#pragma once

#include "common.h"
#include "shared.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

#include "materials/matutils.h"
#include "materials/microfacet.h"
#include "sampling.h"
#include "sampler.h"

KRR_NAMESPACE_BEGIN

using namespace shader;
using namespace bsdf;

class DiffuseBrdf {
public:

	DiffuseBrdf() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, Sampler& sg) {
		DiffuseBrdf bsdf;
		vec3f wo = sd.toLocal(sd.wo);
		bsdf.setup(sd);
		return bsdf.sample(wo, sg);
	}
	
	__both__ void setup(const ShadingData &sd) {
		diffuse = sd.diffuse;
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		return diffuse * M_1_PI;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample;
		vec2f u = sg.get2D();
		sample.wi = cosineSampleHemisphere(u);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		return wi.z * M_1_PI;
	}

	vec3f diffuse;
};

class MicrofacetBrdfAlter {
	// yet another microfacet implementation for comparison.
public:
	MicrofacetBrdfAlter() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, Sampler& sg) {
		MicrofacetBrdfAlter bsdf;
		vec3f wo = sd.toLocal(sd.wo);
		bsdf.setup(sd);
		return bsdf.sample(wo, sg);
	}

	__both__ void setup(const ShadingData & sd) {
		albedo = sd.specular;
		alpha = pow2(sd.roughness);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f h = normalize(wo + wi);
		float woDotH = dot(wo, h);

		float D = evalNdfGGX(alpha, h.z);
		float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
		vec3f F = evalFresnelSchlick(albedo, vec3f(1.f), woDotH);
		return F * D * G * 0.25f / (wo.z * wi.z);
	}

	__both__  BSDFSample sample(vec3f wo, Sampler& sg) const {
		// @ref: Sampling the GGX Distribution of Visible Normals
		BSDFSample sample = {};
		vec2f u = sg.get2D();
		vec3f wi = {};

		vec3f h = sampleGGX_VNDF(alpha, wo, u);    
		wi = Reflect(wo, h);		// Reflect the outgoing direction to find the incident direction.
		
		sample.wi = wi;
		sample.pdf = pdf(wo, wi);
		sample.f = f(wo, wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (min(wo.z, wi.z) < minCosTheta) return 0.f;

		vec3f h = normalize(wo + wi);
		float woDotH = dot(wo, h);
		float pdf = evalPdfGGX_VNDF(alpha, wo, h);
		return pdf / (4.f * woDotH);
	}

	vec3f albedo;		// specular reflectance
	float alpha;		// GGX alpha (r^2)
};

class MicrofacetBrdf {
public:
	MicrofacetBrdf() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, Sampler& sg) {
		MicrofacetBrdf bsdf;
		vec3f wo = sd.toLocal(sd.wo);
		bsdf.setup(sd);
		return bsdf.sample(wo, sg);
	}

	__both__ void setup(const ShadingData& sd) {
		R = sd.specular;
		alpha = GGXMicrofacetDistribution::RoughnessToAlpha(sd.roughness);
		alpha = pow2(sd.roughness);
		distribution.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
		vec3f wh = wi + wo;
		if (cosThetaI == 0 || cosThetaO == 0) return 0;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return 0;
			wh = normalize(wh);
		vec3f F = evalFresnelSchlick(R, vec3f(1.f), dot(wo, wh));
		return distribution.D(wh) * distribution.G(wo, wi) * F * 0.25f /
			(cosThetaI * cosThetaO);
	}

	__both__  BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample = {};
		vec3f wi, wh;
		vec2f u = sg.get2D();

		if (wo.z == 0) return sample;
		wh = distribution.Sample(wo, u);
		if (dot(wo, wh) < 0) return sample;   

		wi = Reflect(wo, wh);
		if (!SameHemisphere(wo, wi)) return sample;

		// Compute PDF of _wi_ for microfacet reflection
		sample.f = f(wo, wi);
		sample.wi = wi;
		sample.pdf = distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
		//printf("F: %.8f, PDF: %.8f\n", sample.f, sample.pdf);
		return sample;

	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		vec3f wh = normalize(wo + wi);
		return distribution.Pdf(wo, wi) / (4 * dot(wo, wh));
	}

	vec3f R;		// specular reflectance
	float alpha;		// GGX alpha (r^2)
	GGXMicrofacetDistribution distribution;
};

class FresnelBlendedBrdf {
public:
	FresnelBlendedBrdf() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData & sd, Sampler & sg) {
		FresnelBlendedBrdf bsdf;
		bsdf.setup(sd);
		vec3f wo = sd.toLocal(sd.wo);
		return bsdf.sample(wo, sg);
	}

	__both__ void setup(const ShadingData & sd) {
		diffuse = sd.diffuse;
		specular = sd.specular;
		//printf("specular: %f %f %f\n", specular.x, specular.y, specular.z);
		float alpha = ggx.RoughnessToAlpha(sd.roughness);
		//printf("roughness: %f, alpha: %f\n", sd.roughness, alpha);
		ggx.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f diff = (28.f / (23.f * M_PI)) * diffuse
			* (vec3f(1.f) - specular)
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wi)))
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wo)));
		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return diff;
		wh = normalize(wh);
		float D = ggx.D(wh);
		vec3f F = evalFresnelSchlick(specular, vec3f(1.f), dot(wi, wh));
		vec3f spec = D * F * 0.25f / (fabs(dot(wi, wh)) * max(AbsCosTheta(wi), AbsCosTheta(wo)));

		//printf("diff: %.8f , spec: %.8f \n", diff.x, spec.x);
		return diff + spec;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler & sg) const {
		BSDFSample sample = {};
		float comp = sg.get1D();
		vec2f u = sg.get2D();
		if (comp < 0.5f) {
			sample.wi = cosineSampleHemisphere(u);
		}
		else {
			vec3f wh = ggx.Sample(wo, u);
			sample.wi = Reflect(wo, wh);
		}
		if (!SameHemisphere(sample.wi, wo)) return sample;
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		float diffPdf = wi.z * M_1_PI;

		vec3f wh = normalize(wo + wi);
		float specPdf = ggx.Pdf(wo, wh) / (4 * dot(wo, wh));

		return 0.5 * (diffPdf + specPdf);
	}
private:
	vec3f diffuse;
	vec3f specular;
	GGXMicrofacetDistribution ggx;
};

class BxDF :public TaggedPointer<DiffuseBrdf, MicrofacetBrdf, FresnelBlendedBrdf, MicrofacetBrdfAlter>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, Sampler& sg, int bsdfIndex) {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sampleInternal(sd, sg); };
		return dispatch(sample, bsdfIndex);
	}

	__both__ inline void setup(const ShadingData &sd) {
		auto setup = [&](auto ptr)->void {return ptr->setup(sd); };
		return dispatch(setup);
	}

	__both__ inline vec3f f(vec3f wo, vec3f wi) const {
		auto f = [&](auto ptr)->vec3f {return ptr->f(wo, wi); };
		return dispatch(f);
	}

	__both__ inline BSDFSample sample(vec3f wo, Sampler& sg) const{
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sample(wo, sg); };
		return dispatch(sample);
	}

	__both__ inline float pdf(vec3f wo, vec3f wi) const {
		auto pdf = [&](auto ptr)->float {return ptr->pdf(wo, wi); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END