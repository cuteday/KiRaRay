#pragma once

#include "common.h"
#include "taggedptr.h"

#include "materials/diffuse.h"
#include "materials/microfacet.h"
#include "materials/fresnelblend.h"
#include "materials/disney.h"
#include "materials/dielectric.h"
#include "materials/principled.h"

KRR_NAMESPACE_BEGIN
using namespace shader;

class BxDF :public TaggedPointer<DiffuseBrdf, 
	FresnelBlendBrdf, 
	DielectricBsdf,
	DisneyBsdf, 
	PrincipledBsdf>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE static BSDFSample sample(const ShadingData& sd, Vector3f wo, Sampler& sg, int bsdfIndex) {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sampleInternal(sd, wo, sg); };
		return dispatch(sample, bsdfIndex);
	}

	KRR_CALLABLE static Color f(const ShadingData& sd, Vector3f wo, Vector3f wi, int bsdfIndex) {
		auto f = [&](auto ptr)->Vector3f {return ptr->fInternal(sd, wo, wi); };
		return dispatch(f, bsdfIndex);
	}

	KRR_CALLABLE static float pdf(const ShadingData& sd, Vector3f wo, Vector3f wi, int bsdfIndex) {
		auto pdf = [&](auto ptr)->float {return ptr->pdfInternal(sd, wo, wi); };
		return dispatch(pdf, bsdfIndex);
	}

	KRR_CALLABLE static BSDFType flags(const ShadingData& sd, int bsdfIndex) {
		auto flags = [&](auto ptr)->BSDFType {return ptr->flagsInternal(sd); };
		return dispatch(flags, bsdfIndex);
	}

	KRR_CALLABLE void setup(const ShadingData &sd) {
		auto setup = [&](auto ptr)->void {return ptr->setup(sd); };
		return dispatch(setup);
	}

	// [NOTE] f the cosine theta term in render equation is not contained in f().
	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		auto f = [&](auto ptr) -> Color { return ptr->f(wo, wi); };
		return dispatch(f);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler& sg) const{
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sample(wo, sg); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		auto pdf = [&](auto ptr)->float {return ptr->pdf(wo, wi); };
		return dispatch(pdf);
	}

	KRR_CALLABLE BSDFType flags() const {
		auto flags = [&](auto ptr) -> BSDFType { return ptr->flags(); };
		return dispatch(flags);
	}
};

KRR_NAMESPACE_END