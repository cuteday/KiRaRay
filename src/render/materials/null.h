#pragma once

#include "common.h"

#include "util/math_utils.h"
#include "sampler.h"

#include "bxdf.h"
#include "matutils.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;

class NullBsdf {
public:
	NullBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(NullBsdf);

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) {}

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
							TransportMode mode = TransportMode::Radiance) const {
		return Spectrum::Zero();
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		BSDFSample sample;
		sample.wi	 = -wo;
		sample.pdf	 = 1;
		sample.f	 = Spectrum::Ones() / AbsCosTheta(wo);
		sample.flags = BSDF_NULL;
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		return 0;
	}

	KRR_CALLABLE BSDFType flags() const { return BSDF_NULL; }
};

KRR_NAMESPACE_END