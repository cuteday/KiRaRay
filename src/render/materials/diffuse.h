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

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
						 TransportMode mode = TransportMode::Radiance) const {
		if (!SameHemisphere(wo, wi)) return Spectrum::Zero();
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

	Spectrum diffuse;
};

KRR_NAMESPACE_END