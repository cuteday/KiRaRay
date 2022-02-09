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
		bsdf.setup(sd);
		return bsdf.sample(sd.wo, sg);
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

// incorrect implementation
class MicrofacetBrdf {
public:
	MicrofacetBrdf() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, Sampler& sg) {
		MicrofacetBrdf bsdf;
		bsdf.setup(sd);
		return bsdf.sample(sd.wo, sg);
	}

	__both__ void setup(const ShadingData & sd) {
		albedo = sd.specular;
		alpha = sd.roughness * sd.roughness;
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f h = normalize(wo + wi);
		float woDotH = dot(wo, h);

		float D = evalNdfGGX(alpha, h.z);
		float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
		vec3f F = evalFresnelSchlick(albedo, 1, woDotH);
		return F * D * G * 0.25f / wo.z;
	}

	__both__  BSDFSample sample(vec3f wo, Sampler& sg) const {
		// @ref: Sampling the GGX Distribution of Visible Normals
		BSDFSample sample = {};
		vec2f u = sg.get2D();
		float pdf;
		vec3f wi = {};

		vec3f h = sampleGGX_VNDF(alpha, wo, u, pdf);    // pdf = G1(wo) * D(h) * max(0,dot(wo,h)) / wo.z

		// Reflect the outgoing direction to find the incident direction.
		float woDotH = dot(wo, h);
		wi = 2.f * woDotH * h - wo;

		float G = evalMaskingSmithGGXSeparable(alpha, wo.z, wi.z);
		float GOverG1wo = evalG1GGX(alpha * alpha, wi.z);

		vec3f F = evalFresnelSchlick(albedo, 1.f, woDotH);
		pdf /= (4.f * woDotH); // Jacobian of the reflection operator.
		vec3f f = F * GOverG1wo / wi.z;

		sample.f = f;
		sample.pdf = pdf;
		sample.wi = wi;
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

class FresnelBlendedBrdf {
public:
	FresnelBlendedBrdf() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData & sd, Sampler & sg) {
		FresnelBlendedBrdf bsdf;
		bsdf.setup(sd);
		return bsdf.sample(sd.wo, sg);
	}

	__both__ void setup(const ShadingData & sd) {
		diffuse = sd.diffuse;
		specular = sd.specular;
		float alpha = ggx.RoughnessToAlpha(sd.roughness);
		ggx.setup(alpha, alpha, false);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f diff = (28.f / (23.f * M_PI)) * diffuse
			 * (vec3f(1.f) - specular) 
			 * (1.f - pow5(1 - .5f * AbsCosTheta(wi)))
			 * (1.f - pow5(1 - .5f * AbsCosTheta(wo)));
		vec3f wh = wi + wo;
		//printf("%.9f\n", diff);
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return diff;
		wh = normalize(wh);
		float D = ggx.D(wh);
		//printf("%f\n", diff);
		vec3f spec = D /
			(4 * fabs(dot(wi, wh)) * max(AbsCosTheta(wi), AbsCosTheta(wo))) *
			evalFresnelSchlick(specular, vec3f(1.f), dot(wi, wh));

		return diff;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler & sg) const {
		BSDFSample sample = {};
		vec2f u = sg.get2D();
		sample.wi = cosineSampleHemisphere(u);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		return wi.z * M_1_PI;
	}
private:
	vec3f diffuse;
	vec3f specular;
	GGXMicrofacetDistribution ggx;
};

class BxDF :public TaggedPointer<DiffuseBrdf, MicrofacetBrdf, FresnelBlendedBrdf>{
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