#pragma once

#include "common.h"
#include "shared.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

struct BSDFSample {
	vec3f f;
	vec3f wi;
	float pdf = 0;
	uint flags;
};

class DiffuseBxDF {
public:

	DiffuseBxDF() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, vec2f u) {
		DiffuseBxDF bsdf;
		bsdf.setup(sd);
		return bsdf.sample(sd.wo, u);
	}

	__both__ DiffuseBxDF(vec3f r) : diffuse(r) {}
	
	__both__ void setup(const ShadingData &sd) {
		diffuse = sd.diffuse;
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		return diffuse * M_1_PI * wi.z;
	}

	__both__ BSDFSample sample(vec3f wo, vec2f u) const {
		BSDFSample sample;
		sample.wi = utils::cosineSampleHemisphere(u);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		return wi.z * M_1_PI;
	}

private:
	vec3f diffuse;
};

class MicrofacetBxDF {
public:

	MicrofacetBxDF() = default;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, vec2f u) {
		MicrofacetBxDF bsdf;
		bsdf.setup(sd);
		return bsdf.sample(sd.wo, u);
	}

	__both__ void setup(const ShadingData & sd) {
		diffuse = sd.diffuse;
		return void();
	}

	__both__ vec3f f(vec3f wo, vec3f wi) const {
		return diffuse * M_1_PI * wi.z;
	}

	__both__  BSDFSample sample(vec3f wo, vec2f u) const {
		BSDFSample sample;
		sample.wi = utils::cosineSampleHemisphere(u);
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	__both__ float pdf(vec3f wo, vec3f wi) const {
		return wi.z * M_1_PI;
	}

private:
	vec3f diffuse;
};


class BxDF :public TaggedPointer<DiffuseBxDF, MicrofacetBxDF>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline static BSDFSample sampleInternal(const ShadingData& sd, vec2f u, int index) {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sampleInternal(sd, u); };
		return dispatch(sample, index);
	}

	__both__ inline void setup(const ShadingData &sd) {
		auto setup = [&](auto ptr)->void {return ptr->setup(sd); };
		return dispatch(setup);
	}

	__both__ inline vec3f f(vec3f wo, vec3f wi) const {
		auto f = [&](auto ptr)->vec3f {return ptr->f(wo, wi); };
		return dispatch(f);
	}

	__both__ inline BSDFSample sample(vec3f wo, vec2f u) const{
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sample(wo, u); };
		return dispatch(sample);
	}

	__both__ inline float pdf(vec3f wo, vec3f wi) const {
		auto pdf = [&](auto ptr)->float {return ptr->pdf(wo, wi); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END