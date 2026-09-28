#pragma once

#include "openpbr/model.h"
#include "render/shared.h"
#include "sampler.h"

NAMESPACE_BEGIN(krr)

KRR_CALLABLE KRR_NOINLINE void evaluateAuthoredFlags(BSDFData &data,
													 const rt::MaterialProgramData &program,
													 const MaterialContext &context) {
	auto values				  = program.evaluate(context, rt::MaterialEvaluation::Classification);
	data.metallic			  = values[MaterialParameter::Metalness][0];
	data.specularTransmission = values[MaterialParameter::TransmissionWeight][0];
}

KRR_CALLABLE void prepareAuthoredFlags(BSDFData &data, const rt::MaterialProgramData &program,
									   const MaterialContext &context) {
	data.metallic			  = program.defaults[MaterialParameter::Metalness][0];
	data.specularTransmission = program.defaults[MaterialParameter::TransmissionWeight][0];
	if (program.classification.count) evaluateAuthoredFlags(data, program, context);
}

class OpenPbrBsdf {
public:
	_DEFINE_BSDF_INTERNAL_ROUTINES(OpenPbrBsdf);

	KRR_CALLABLE void setup(const SurfaceInteraction &interaction) {
		intr	 = &interaction;
		prepared = false;
	}

	KRR_CALLABLE KRR_NOINLINE const openpbr::Model &prepare(Vector3f wo) const {
		if (prepared && wo[0] == outgoing[0] && wo[1] == outgoing[1] && wo[2] == outgoing[2])
			return model;
		MaterialContext context;
		context.uv		  = {intr->uv[0], intr->uv[1], 0};
		context.normal	  = {intr->n[0], intr->n[1], intr->n[2]};
		context.tangent	  = {intr->tangent[0], intr->tangent[1], intr->tangent[2]};
		context.bitangent = {intr->bitangent[0], intr->bitangent[1], intr->bitangent[2]};
		auto values		  = intr->material->mProgram.evaluate(context);
		model	 = openpbr::Model(values, context, intr->toWorld(wo), intr->lambda,
								  *intr->material->getColorSpace(), intr->material->mProgram.model);
		outgoing = wo;
		prepared = true;
		return model;
	}

	KRR_CALLABLE KRR_NOINLINE Spectrum f(Vector3f wo, Vector3f wi,
										 TransportMode mode = TransportMode::Radiance) const {
		if (wi[2] == 0) return Spectrum(0);
		return prepare(wo).evalCos(intr->toWorld(wi), mode) / fabsf(wi[2]);
	}

	KRR_CALLABLE KRR_NOINLINE float pdf(Vector3f wo, Vector3f wi,
										TransportMode mode = TransportMode::Radiance) const {
		return prepare(wo).pdf(intr->toWorld(wi));
	}

	KRR_CALLABLE KRR_NOINLINE BSDFEval eval(Vector3f wo, Vector3f wi,
										  TransportMode mode = TransportMode::Radiance) const {
		const auto &model = prepare(wo);
		Vector3f direction = intr->toWorld(wi);
		// Authored normals can give tangent directions a nonzero sampling density.
		if (wi[2] == 0) return {Spectrum(0), model.pdf(direction)};
		BSDFEval result = model.evalCosPdf(direction, mode);
		result.f /= fabsf(wi[2]);
		return result;
	}

	KRR_CALLABLE KRR_NOINLINE BSDFSample
	sample(Vector3f wo, Sampler &sampler, TransportMode mode = TransportMode::Radiance) const {
		const auto &model  = prepare(wo);
		Vector3f direction = model.sample(sampler.get1D(), sampler.get2D());
		if (direction.squaredNorm() == 0) return {};
		Vector3f wi	  = intr->toLocal(direction);
		if (wi[2] == 0) return {};
		BSDFEval result = model.evalCosPdf(direction, mode);
		if (result.pdf <= 0) return {};
		return {result.f / fabsf(wi[2]), wi, result.pdf,
				wi[2] * wo[2] > 0 ? BSDF_GLOSSY_REFLECTION : BSDF_GLOSSY_TRANSMISSION};
	}
	
	KRR_CALLABLE BSDFType flags() const { return intr->sd.getBsdfType(); }
	const SurfaceInteraction *intr{nullptr};

private:
	mutable openpbr::Model model;
	mutable Vector3f outgoing{0};
	mutable bool prepared{false};
};

class PreviewSurfaceBsdf : public OpenPbrBsdf {
public:
	_DEFINE_BSDF_INTERNAL_ROUTINES(PreviewSurfaceBsdf);
};

NAMESPACE_END(krr)
