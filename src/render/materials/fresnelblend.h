#pragma once

#include "common.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

#include "bxdf.h"
#include "matutils.h"
#include "fresnel.h"
#include "microfacet.h"
#include "sampling.h"
#include "sampler.h"

KRR_NAMESPACE_BEGIN
using namespace shader;

class FresnelBlendBrdf {
public:
	FresnelBlendBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(FresnelBlendBrdf);

	__both__ void setup(const ShadingData & sd) {
		diffuse = sd.diffuse;
		specular = sd.specular;
		float alpha = ggx.RoughnessToAlpha(sd.roughness);
		alpha = sd.roughness * sd.roughness;
		ggx.setup(alpha, alpha, true);
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		vec3f diff = (28.f / (23.f * M_PI)) * diffuse
			* (vec3f(1) - specular)
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wi)))
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wo)));
		vec3f wh = wi + wo;
		if (!any(wh)) return diff;
		wh = normalize(wh);
		float D = ggx.D(wh);
		vec3f F = FrSchlick(specular, vec3f(1.f), dot(wi, wh));
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

KRR_NAMESPACE_END