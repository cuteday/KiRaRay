#pragma once

#include "common.h"
#include "math/math.h"
#include "math/utils.h"
#include "sampler.h"

#include "bxdf.h"
#include "matutils.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN
using namespace shader;
using namespace bsdf;

class DiffuseBrdf {
public:
	DiffuseBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(DiffuseBrdf);

	__both__ void setup(const ShadingData& sd) {
		diffuse = sd.diffuse;
	}

	__both__ Color f(vec3f wo, vec3f wi) const {
		return diffuse * M_INV_PI;
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
		return fabs(wi[2]) * M_INV_PI;
	}

	Color diffuse;
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

	__both__ Color f(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi)) return reflection * M_INV_PI;
		return transmission * M_INV_PI;
	}

	__both__ BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample;
		float c = sg.get1D();
		vec2f u = sg.get2D();
		vec3f wi = cosineSampleHemisphere(u);
		if (c < pR) {
			if (!SameHemisphere(wi, wo))
				wi[2] *= -1;
		}
		else {
			if (SameHemisphere(wi, wo))
				wi[2] *= -1;
		}
		sample.wi = wi;
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		if (SameHemisphere(wo, wi))
			return pR * fabs(wi[2]);
		return pT * fabs(wi[2]);
	}

	Color reflection{ 0 }, transmission{ 0 };
	float pR{ 1 }, pT{ 0 };
};


KRR_NAMESPACE_END