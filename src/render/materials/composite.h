#pragma once

#include "diffuse.h"
#include "conductor.h"
#include "dielectric.h"
#include "openpbr.h"

NAMESPACE_BEGIN(krr)

KRR_CALLABLE KRR_NOINLINE BSDFType authoredComponentFlags(
	const rt::MaterialProgramData &program, const MaterialContext &context) {
	using P = MaterialParameter;
	auto values = program.evaluate(context, rt::MaterialEvaluation::Classification);
	if (values[P::Weight][0] <= 0) return BSDF_UNSET;
	float alpha = pow2(values[P::SpecularRoughness][0]);
	switch (program.model) {
		case MaterialModel::Diffuse: return BSDF_DIFFUSE_REFLECTION;
		case MaterialModel::Conductor: {
			float aspect = sqrtf(fmaxf(.1f, 1 - .9f * values[P::SpecularAnisotropy][0]));
			return BSDF_REFLECTION | (fmaxf(alpha / aspect, alpha * aspect) <= .001f
				? BSDF_SPECULAR : BSDF_GLOSSY);
		}
		case MaterialModel::Dielectric:
			return BSDF_REFLECTION | BSDF_TRANSMISSION |
				(alpha <= .001f || values[P::SpecularIor][0] == 1 ? BSDF_SPECULAR : BSDF_GLOSSY);
		case MaterialModel::OpenPBR:
		case MaterialModel::PreviewSurface:
			return BSDF_SMOOTH | BSDF_REFLECTION | BSDF_TRANSMISSION;
		default: return BSDF_UNSET;
	}
}

KRR_CALLABLE KRR_NOINLINE void prepareCompositeFlags(BSDFData &data,
	const rt::MaterialProgramData &program, const MaterialContext &context) {
	data.authoredFlags = BSDF_UNSET;
	if (program.model != MaterialModel::Composite) {
		data.authoredFlags = authoredComponentFlags(program, context);
		return;
	}
	for (uint32_t index = 0; index < program.componentCount; ++index)
		data.authoredFlags |= authoredComponentFlags(program.components[index], context);
}

class CompositeBsdf {
public:
	_DEFINE_BSDF_INTERNAL_ROUTINES(CompositeBsdf);

	KRR_CALLABLE void setup(const SurfaceInteraction &interaction) {
		intr = &interaction;
		context.uv = {intr->uv[0], intr->uv[1], 0};
		context.normal = {intr->n[0], intr->n[1], intr->n[2]};
		context.tangent = {intr->tangent[0], intr->tangent[1], intr->tangent[2]};
		context.bitangent = {intr->bitangent[0], intr->bitangent[1], intr->bitangent[2]};
		const auto &program = intr->material->mProgram;
		count = program.model == MaterialModel::Composite ? program.componentCount : 1;
		sum = 0;
		for (uint32_t index = 0; index < count; ++index) {
			const auto &leaf = component(index);
			float weight = leaf.defaults[MaterialParameter::Weight][0];
			if (leaf.weight.count || leaf.kind == MaterialProgramKind::Simple)
				weight = leaf.evaluate(context, rt::MaterialEvaluation::Weight)[MaterialParameter::Weight][0];
			weights[index] = fmaxf(0, weight);
			sum += weights[index];
		}
	}

	KRR_CALLABLE BSDFEval eval(Vector3f wo, Vector3f wi,
		TransportMode mode = TransportMode::Radiance) const {
		BSDFEval result;
		if (sum <= 0) return result;
		Vector3f outgoing = intr->toWorld(wo), incoming = intr->toWorld(wi);
		for (uint32_t index = 0; index < count; ++index) {
			if (weights[index] == 0) continue;
			BSDFEval value = evaluate(component(index), outgoing, incoming, mode);
			result.f += weights[index] * value.f;
			result.pdf += weights[index] / sum * value.pdf;
		}
		result.f = wi[2] == 0 ? Spectrum(0) : Spectrum(result.f / fabsf(wi[2]));
		return result;
	}

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
		TransportMode mode = TransportMode::Radiance) const { return eval(wo, wi, mode).f; }
	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
		TransportMode mode = TransportMode::Radiance) const { return eval(wo, wi, mode).pdf; }

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sampler,
		TransportMode mode = TransportMode::Radiance) const {
		if (sum <= 0) return {};
		uint32_t selected = 0;
		float target = count == 1 ? 0 : sampler.get1D() * sum;
		for (uint32_t index = 0; index < count; ++index) {
			if (weights[index] <= 0) continue;
			selected = index;
			if (target < weights[index]) break;
			target -= weights[index];
		}
		Vector3f outgoing = intr->toWorld(wo);
		BSDFSample result = sampleComponent(component(selected), outgoing, sampler, mode);
		if (result.pdf <= 0) return {};
		Vector3f incoming = result.wi;
		result.wi = intr->toLocal(incoming);
		if (result.wi[2] == 0) return {};
		result.f *= weights[selected];
		result.pdf *= weights[selected] / sum;
		// Delta samples carry discrete mass; ordinary eval() cannot reconstruct them.
		if (!result.isDelta()) {
			for (uint32_t index = 0; index < count; ++index) {
				if (index == selected || weights[index] == 0) continue;
				BSDFEval value = evaluate(component(index), outgoing, incoming, mode);
				result.f += weights[index] * value.f;
				result.pdf += weights[index] / sum * value.pdf;
			}
		}
		result.f /= fabsf(result.wi[2]);
		return result;
	}

	KRR_CALLABLE BSDFType flags() const { return intr->sd.authoredFlags; }

private:
	KRR_CALLABLE const rt::MaterialProgramData &component(uint32_t index) const {
		const auto &program = intr->material->mProgram;
		return program.model == MaterialModel::Composite ? program.components[index] : program;
	}
	KRR_CALLABLE Frame frame(const MaterialValues &values) const {
		Vector3f normal = openpbr::unit(openpbr::vector(values[MaterialParameter::Normal]), intr->n);
		return openpbr::frame(normal, openpbr::vector(values[MaterialParameter::Tangent]),
			intr->bitangent, normal);
	}
	KRR_CALLABLE Spectrum color(const MaterialValues &values) const {
		auto value = values[MaterialParameter::BaseColor];
		return Spectrum::fromRGB(RGB(value[0], value[1], value[2]).cwiseMax(0.f).cwiseMin(1.f),
			SpectrumType::RGBBounded, intr->lambda, *intr->material->getColorSpace());
	}
	KRR_CALLABLE ConductorBsdf conductor(const MaterialValues &values) const {
		float alpha = pow2(values[MaterialParameter::SpecularRoughness][0]);
		float aspect = sqrtf(fmaxf(.1f, 1 - .9f * values[MaterialParameter::SpecularAnisotropy][0]));
		return ConductorBsdf(color(values), alpha / aspect, alpha * aspect);
	}
	KRR_CALLABLE DielectricBsdf dielectric(const MaterialValues &values) const {
		float alpha = pow2(values[MaterialParameter::SpecularRoughness][0]);
		return DielectricBsdf(color(values), fmaxf(.01f, values[MaterialParameter::SpecularIor][0]),
			alpha, alpha);
	}
	KRR_CALLABLE KRR_NOINLINE BSDFEval evaluate(const rt::MaterialProgramData &program,
		Vector3f wo, Vector3f wi, TransportMode mode) const {
		auto values = program.evaluate(context);
		if (program.model == MaterialModel::OpenPBR || program.model == MaterialModel::PreviewSurface) {
			openpbr::Model model(values, context, wo, intr->lambda,
				*intr->material->getColorSpace(), program.model);
			return model.evalCosPdf(wi, mode);
		}
		Frame basis = frame(values);
		wo = basis.toLocal(wo);
		wi = basis.toLocal(wi);
		BSDFEval result;
		switch (program.model) {
			case MaterialModel::Diffuse: {
				DiffuseBrdf bsdf;
				bsdf.diffuse = color(values);
				result = bsdf.eval(wo, wi, mode);
				break;
			}
			case MaterialModel::Conductor: result = conductor(values).eval(wo, wi, mode); break;
			case MaterialModel::Dielectric: result = dielectric(values).eval(wo, wi, mode); break;
			default: break;
		}
		result.f *= fabsf(wi[2]);
		return result;
	}
	KRR_CALLABLE KRR_NOINLINE BSDFSample sampleComponent(const rt::MaterialProgramData &program,
		Vector3f wo, Sampler &sampler, TransportMode mode) const {
		auto values = program.evaluate(context);
		if (program.model == MaterialModel::OpenPBR || program.model == MaterialModel::PreviewSurface) {
			openpbr::Model model(values, context, wo, intr->lambda,
				*intr->material->getColorSpace(), program.model);
			Vector3f wi = model.sample(sampler.get1D(), sampler.get2D());
			if (wi.squaredNorm() == 0) return {};
			BSDFEval value = model.evalCosPdf(wi, mode);
			BSDFType flags = dot(wo, intr->n) * dot(wi, intr->n) > 0
				? BSDF_GLOSSY_REFLECTION : BSDF_GLOSSY_TRANSMISSION;
			return {value.f, wi, value.pdf, flags};
		}
		Frame basis = frame(values);
		wo = basis.toLocal(wo);
		BSDFSample result;
		switch (program.model) {
			case MaterialModel::Diffuse: {
				DiffuseBrdf bsdf;
				bsdf.diffuse = color(values);
				result = bsdf.sample(wo, sampler, mode);
				break;
			}
			case MaterialModel::Conductor: result = conductor(values).sample(wo, sampler, mode); break;
			case MaterialModel::Dielectric: result = dielectric(values).sample(wo, sampler, mode); break;
			default: return {};
		}
		if (result.pdf <= 0) return {};
		result.f *= fabsf(result.wi[2]);
		result.wi = basis.toWorld(result.wi);
		return result;
	}

	const SurfaceInteraction *intr{nullptr};
	MaterialContext context;
	float weights[MaterialComponentLimit]{};
	float sum{0};
	uint32_t count{0};
};

NAMESPACE_END(krr)
