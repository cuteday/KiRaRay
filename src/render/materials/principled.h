#pragma once

#include "common.h"
#include "math/math.h"
#include "math/utils.h"
#include "sampler.h"

#include "bxdf.h"
#include "dielectric.h"
#include "matutils.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;
using namespace shader;

class PrincipledBsdf {
public:
	_DEFINE_BSDF_INTERNAL_ROUTINES(PrincipledBsdf);

	KRR_CALLABLE void setup(const ShadingData &sd) { 
	
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		return 0;
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg) const {
		BSDFSample sample{};
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		return 0;
	}

	KRR_CALLABLE BSDFType flags() const { 
		return BSDF_UNSET; 
	}

private:

};

KRR_NAMESPACE_END