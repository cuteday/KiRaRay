#pragma once

#include "common.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

#include "materials/bxdf.h"
#include "materials/matutils.h"
#include "materials/diffuse.h"
#include "materials/microfacet.h"
#include "materials/fresnel.h"
#include "materials/fresnelblend.h"
#include "materials/disney.h"
#include "materials/falcor.h"
#include "sampling.h"
#include "sampler.h"


KRR_NAMESPACE_BEGIN
using namespace shader;

class BxDF :public TaggedPointer<
	DiffuseBrdf, 
	MicrofacetBrdf, 
	FresnelBlendBrdf, 
	UniformBsdf,
	DisneyBsdf>{
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