#pragma once

#include "common.h"
#include "shared.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

#include "materials/matutils.h"
#include "materials/microfacet.h"
#include "materials/fresnel.h"
#include "sampling.h"
#include "sampler.h"

#define SCHLICK_APPROX_FRESNEL 1

KRR_NAMESPACE_BEGIN

#define _DEFINE_BSDF_INTERNAL_ROUTINES(bsdf_name)														\
	__both__ inline static BSDFSample sampleInternal(const ShadingData &sd, vec3f wo, Sampler & sg) {	\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.sample(wo, sg);																		\
	}																									\
																										\
	__both__ inline static vec3f fInternal(const ShadingData& sd, vec3f wo, vec3f wi) {					\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.f(wo, wi);																			\
	}																									\
																										\
	__both__ inline static float pdfInternal(const ShadingData& sd, vec3f wo, vec3f wi) {				\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.pdf(wo, wi);																		\
	}																									\

using namespace shader;
//using namespace bsdf;

class DiffuseBrdf {
public:

	DiffuseBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(DiffuseBrdf);
	
	__both__ void setup(const ShadingData &sd) {
		diffuse = sd.diffuse;
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		return diffuse * M_1_PI;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample;
		vec2f u = sg.get2D();
		vec3f wi = cosineSampleHemisphere(u);
		sample.wi = ToSameHemisphere(wi, wo);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		return fabs(wi.z) * M_1_PI;
	}

	vec3f diffuse;
};

class DiffuseBsdf {
public:

	DiffuseBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(DiffuseBsdf);

	__both__ void setup(const ShadingData& sd) {
		// luminance as weight to sample different components?
		reflection = sd.diffuse;
		transmission = sd.transmission;
		if (any(reflection) || any(transmission)) {
			pR = luminance(reflection) / luminance(reflection) + luminance(transmission);
			pT = luminance(transmission) / luminance(reflection) + luminance(transmission);
		}
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi)) return reflection * M_1_PI;
		return transmission * M_1_PI;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample;
		float c = sg.get1D();
		vec2f u = sg.get2D();
		vec3f wi = cosineSampleHemisphere(u);
		if (c < pR) {
			if (!SameHemisphere(wi, wo))
				wi.z *= -1;
		}
		else {
			if (SameHemisphere(wi, wo))
				wi.z *= -1;
		}
		sample.wi = wi;
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi))
			return pR * fabs(wi.z);
		else return pT * fabs(wi.z);
		return wi.z * M_1_PI;
	}

	vec3f reflection{ 0 }, transmission{ 0 };
	float pR{ 1 }, pT{ 0 };
};

class MicrofacetBrdf {
public:
	MicrofacetBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(MicrofacetBrdf);

	__both__ void setup(const ShadingData& sd) {
		R = sd.specular;
		float alpha = GGXMicrofacetDistribution::RoughnessToAlpha(sd.roughness);
		alpha = pow2(sd.roughness);
		eta = sd.IoR;
		distribution.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
		vec3f wh = wi + wo;
		if (cosThetaI == 0 || cosThetaO == 0) return 0;
		if (!any(wh)) return 0;
		wh = normalize(wh);
		// fresnel is also on the microfacet (wrt to wh)
#if SCHLICK_APPROX_FRESNEL
		vec3f F = evalFresnelSchlick(R, vec3f(1.f), dot(wo, wh));
#else
		vec3f F = R * FrDielectric(dot(wo, wh), wo.z > 0 ? eta : 1 / eta);	// etaT / etaI
#endif
		return distribution.D(wh) * distribution.G(wo, wi) * F /
			(4.f * cosThetaI * cosThetaO);
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
		return sample;

	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		vec3f wh = normalize(wo + wi);
		return distribution.Pdf(wo, wi) / (4 * dot(wo, wh));
	}

	vec3f R;									// specular reflectance
	float eta;									// 
	GGXMicrofacetDistribution distribution;
};

// Microfacet transmission, for dielectric only now.
class MicrofacetBtdf {
public:
	MicrofacetBtdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(MicrofacetBtdf);

	__both__ void setup(const ShadingData& sd) {
		T = sd.transmission;
		float alpha = GGXMicrofacetDistribution::RoughnessToAlpha(sd.roughness);
		alpha = pow2(sd.roughness);
		// TODO: ETA=1 causes NaN, should switch to delta scattering
		etaA = 1; etaB = max(1.01f, sd.IoR);
		etaB = 1.5;
		distribution.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi)) return 0;

		float cosThetaO = wo.z, cosThetaI = wi.z;
		if (cosThetaI == 0 || cosThetaO == 0) return 0;

		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		float eta = CosTheta(wo) > 0 ? etaB/etaA : etaA/etaB;
		vec3f wh = normalize(wo + wi * eta);
		if (wh.z < 0) wh = -wh;

		// Same side?
		if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
		vec3f F = FrDielectric(dot(wo, wh), etaB/etaA);

		float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
		float factor =  1.f / eta;

		return (vec3f(1) - F) * T *
			fabs(distribution.D(wh) * distribution.G(wo, wi) * eta * eta *
				AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
				(cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
	}

	__both__  BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample = {};
		if (wo.z == 0) return sample;
		
		vec2f u = sg.get2D();
		vec3f wh = distribution.Sample(wo, u);
		if (dot(wo, wh) < 0)
			return sample;  // Should be rare
		
		float eta = CosTheta(wo) > 0 ? etaA/etaB : etaB/etaA;
		if (!Refract(wo, wh, eta, &sample.wi)) return sample;

		sample.pdf = pdf(wo, sample.wi);
		sample.f = f(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi)) return 0;
		float eta = CosTheta(wo) > 0 ? etaB/etaA : etaA/etaB;	// wo is outside?
		// wh = wo + eta * wi, eta = etaI / etaT
		vec3f wh = normalize(wo + wi * eta);	// eta=etaI/etaT

		if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
		// Compute change of variables _dwh\_dwi_ for microfacet transmission
		float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
		float dwh_dwi = fabs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
		float pdf = distribution.Pdf(wo, wh);
		//printf("wh pdf: %.6f, dwh_dwi: %.6f\n", pdf, dwh_dwi);
		return pdf * dwh_dwi;
	}

	vec3f T{ 0 };									// specular reflectance
	float etaA, etaB				/* etaA: outside IoR, etaB: inside IoR */;
	GGXMicrofacetDistribution distribution;
};

class FresnelBlendedBrdf {
public:
	FresnelBlendedBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(FresnelBlendedBrdf);

	__both__ void setup(const ShadingData & sd) {
		diffuse = sd.diffuse;
		specular = sd.specular;
		float alpha = ggx.RoughnessToAlpha(sd.roughness);
		alpha = sd.roughness * sd.roughness;
		ggx.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f diff = (28.f / (23.f * M_PI)) * diffuse
			* (vec3f(1) - specular)
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wi)))
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wo)));
		vec3f wh = wi + wo;
		if (!any(wh)) return diff;
		wh = normalize(wh);
		float D = ggx.D(wh);
		vec3f F = evalFresnelSchlick(specular, vec3f(1.f), dot(wi, wh));
		vec3f spec = D * F * 0.25f / (fabs(dot(wi, wh)) * max(AbsCosTheta(wi), AbsCosTheta(wo)));

		return diff + spec;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler & sg) const {
		BSDFSample sample = {};
		float comp = sg.get1D();
		vec2f u = sg.get2D();
		vec3f wi;
		if (comp < 0.5f) {
			wi = cosineSampleHemisphere(u);
			wi = ToSameHemisphere(wi, wo);
		}
		else {
			vec3f wh = ggx.Sample(wo, u);
			wi = Reflect(wo, wh);
			if (!SameHemisphere(wi, wo)) return sample;
		}
		sample.wi = wi;
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		float diffPdf = fabs(wi.z) * M_1_PI;

		vec3f wh = normalize(wo + wi);
		float specPdf = ggx.Pdf(wo, wh) / (4 * dot(wo, wh));

		return 0.5f * (diffPdf + specPdf);
	}

private:
	vec3f diffuse;
	vec3f specular;
	GGXMicrofacetDistribution ggx;
};


class UniformBsdf {
	// A BSDF model taken from NVIDIA's render framework Falcor.
	// Hmm... seems not energy conservative?
public:
	UniformBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(UniformBsdf);

	__both__ void setup(const ShadingData & sd) {
		specTrans = sd.specularTransmission;

		diffuseBsdf.setup(sd);
		microfacetBrdf.setup(sd);
		microfacetBtdf.setup(sd);

		float metallic = getMetallic(sd.diffuse, sd.specular);

		float metallicBRDF = metallic;
		float specularBSDF = (1.f - metallic) * luminance(specTrans);
		float dielectricBSDF = (1.f - metallic) * (1.f - luminance(specTrans));

		float diffuseWeight = luminance(sd.diffuse);
		float specularWeight = luminance(evalFresnelSchlick(sd.specular, 1.f, dot(sd.wo, sd.N)));
	
		pDiff = diffuseWeight * dielectricBSDF;
		pSpecRefl = specularWeight * (metallicBRDF + dielectricBSDF);
		pSpecTrans = any(specTrans) ? specularBSDF : 0;

		float norm = pDiff + pSpecRefl + pSpecTrans;
		if (norm > 0) pDiff /= norm, pSpecRefl /= norm, pSpecTrans /= norm;
		if (pSpecTrans > 0) {
			//printf("pDiff %f pSpecRefl %f pSpecTrans %f\n", pDiff, pSpecRefl, pSpecTrans);
			pDiff = pSpecRefl = pSpecTrans = 1.f / 3;
		}
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		vec3f val = 0;
		if(pDiff) val += (vec3f(1) - specTrans) * diffuseBsdf.f(wo, wi);
		if(pSpecRefl) val += (vec3f(1) - specTrans) * microfacetBrdf.f(wo, wi);
		if(pSpecTrans) val += specTrans * microfacetBtdf.f(wo, wi);
		return val;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler & sg) const {
		float u = sg.get1D();
		BSDFSample sample = {};
		if (u < pDiff) {
			sample = diffuseBsdf.sample(wo, sg);
			sample.f *= (vec3f(1) - specTrans) / pDiff;
			sample.pdf *= pDiff;
			if (pSpecRefl) sample.pdf += pSpecRefl * microfacetBrdf.pdf(wo, sample.wi);
			if (pSpecTrans) sample.pdf += pSpecTrans * microfacetBtdf.pdf(wo, sample.wi);
		}
		else if (u < pDiff + pSpecRefl) {
			sample = microfacetBrdf.sample(wo, sg);
			sample.f *= (vec3f(1) - specTrans) / pSpecRefl;
			sample.pdf *= pSpecRefl;
			if (pDiff) sample.pdf += pDiff * diffuseBsdf.pdf(wo, sample.wi);
			if (pSpecTrans) sample.pdf += pSpecTrans * microfacetBtdf.pdf(wo, sample.wi);
		}
		else if (pSpecTrans > 0) {
			sample = microfacetBtdf.sample(wo, sg);
			//sample.f *= specTrans / pSpecTrans;
			sample.pdf *= pSpecTrans;
			if (pDiff) sample.pdf += pDiff * diffuseBsdf.pdf(wo, sample.wi);
			if (pSpecRefl) sample.pdf += pSpecRefl * microfacetBrdf.pdf(wo, sample.wi);
		}
		//sample.f = f(wo, sample.wi);
		//sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		float val = 0;
		if (pDiff) val += pDiff * diffuseBsdf.pdf(wo, wi);
		if (pSpecRefl) val += pSpecRefl * microfacetBrdf.pdf(wo, wi);
		if (pSpecTrans) val += pSpecTrans * microfacetBtdf.pdf(wo, wi);
		return val;
	}

	float pDiff{ 1 };
	float pSpecTrans{ 0 };
	float pSpecRefl{ 1 };

	vec3f specTrans{ 0 };

	DiffuseBrdf diffuseBsdf;
	MicrofacetBrdf microfacetBrdf;
	MicrofacetBtdf microfacetBtdf;

private:
	__both__ static inline float getMetallic(vec3f diffuse, vec3f spec){
		// This is based on the way that UE4 and Substance Painter 2 converts base+metallic+specular level to diffuse/spec colors
		// We don't have the specular level information, so the assumption is that it is equal to 0.5 (based on the UE4 documentation)
		float d = luminance(diffuse);
		float s = luminance(spec);
		if (s == 0) return 0;
		float b = s + d - 0.08;
		float c = 0.04 - s;
		float root = sqrt(b * b - 0.16 * c);
		float m = (root - b) * 12.5;
		return max(0.f, m);
	}
};

class BxDF :public TaggedPointer<
	DiffuseBrdf, 
	MicrofacetBrdf, 
	FresnelBlendedBrdf, 
	UniformBsdf>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline static BSDFSample sample(const ShadingData& sd, vec3f wo, Sampler& sg, int bsdfIndex) {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sampleInternal(sd, wo, sg); };
		return dispatch(sample, bsdfIndex);
	}

	__both__ inline static vec3f f(const ShadingData& sd, vec3f wo, vec3f wi, int bsdfIndex) {
		auto f = [&](auto ptr)->vec3f {return ptr->fInternal(sd, wo, wi); };
		return dispatch(f, bsdfIndex);
	}

	__both__ inline static float pdf(const ShadingData& sd, vec3f wo, vec3f wi, int bsdfIndex) {
		auto pdf = [&](auto ptr)->float {return ptr->pdfInternal(sd, wo, wi); };
		return dispatch(pdf, bsdfIndex);
	}

	__both__ inline void setup(const ShadingData &sd) {
		auto setup = [&](auto ptr)->void {return ptr->setup(sd); };
		return dispatch(setup);
	}

	// [NOTE] f the cosine theta term in render equation is not contained in f().
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