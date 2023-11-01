#pragma once

#include "common.h"

#include "util/math_utils.h"
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

	KRR_CALLABLE void setup(const SurfaceInteraction& intr) {
		diffuse = intr.sd.diffuse;
	}

	KRR_CALLABLE SampledSpectrum f(Vector3f wo, Vector3f wi,
						 TransportMode mode = TransportMode::Radiance) const {
		if (!SameHemisphere(wo, wi)) return SampledSpectrum::Zero();
		return diffuse * M_INV_PI;
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		BSDFSample sample;
		Vector2f u = sg.get2D();
		Vector3f wi = cosineSampleHemisphere(u);
		sample.wi = ToSameHemisphere(wi, wo);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		sample.flags = BSDF_DIFFUSE_REFLECTION;
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		if (!SameHemisphere(wo, wi)) return 0;
		return fabs(wi[2]) * M_INV_PI;
	}
	
	KRR_CALLABLE BSDFType flags() const {
		return diffuse.any() ? BSDF_DIFFUSE : BSDF_UNSET;
	}

	SampledSpectrum diffuse;
};

class DiffuseBsdf {
public:

	DiffuseBsdf() = default;

	KRR_CALLABLE void setup(const SurfaceInteraction& intr) {
		// luminance as weight to sample different components?
		reflection = (1 - intr.sd.specularTransmission) * intr.sd.diffuse;
		transmission = intr.sd.specularTransmission * intr.sd.diffuse;
		if (any(reflection) || any(transmission)) {
			pR = luminance(reflection) / luminance(reflection) + luminance(transmission);
			pT = luminance(transmission) / luminance(reflection) + luminance(transmission);
		}
	}

	KRR_CALLABLE SampledSpectrum f(Vector3f wo, Vector3f wi) const {
		if (SameHemisphere(wo, wi)) return reflection * M_INV_PI;
		return transmission * M_INV_PI;
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler& sg) const {
		BSDFSample sample;
		float c = sg.get1D();
		Vector2f u = sg.get2D();
		Vector3f wi = cosineSampleHemisphere(u);
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

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		if (SameHemisphere(wo, wi))
			return pR * fabs(wi[2]);
		return pT * fabs(wi[2]);
	}

	SampledSpectrum reflection{ 0 }, transmission{ 0 };
	float pR{ 1 }, pT{ 0 };
};


KRR_NAMESPACE_END