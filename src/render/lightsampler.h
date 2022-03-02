#pragma once

#include "common.h"
#include "taggedptr.h"
#include "light.h"
#include "device/memory.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

struct SampledLight{
	Light light;
	float pdf;
};

class UniformLightSampler{
public:
	UniformLightSampler() = default;

	UniformLightSampler(inter::span<Light> lights):
		// use IterFirst and IterLast to initialize 
		mLights(lights.begin(), lights.end()) {}

	// assumes u in [0, 1)
	__both__ inline SampledLight sample(float u)const {
		SampledLight sl = {};
		uint sampleId = u * mLights.size();
		sl.light = mLights[sampleId];
		sl.pdf = 1.f / mLights.size();
		//DCHECK(0 <= sampleId && sampleId < mLights.size());
		DCHECK(sl.light.ptr());
		return sl;
	}

	__both__ inline float pdf(const Light& light)const {
		return 1.f / mLights.size();
	}

	__both__ inline inter::vector<Light> getLights() {
		return mLights;
	}

private:
	inter::vector<Light> mLights;
};

class LightSampler: TaggedPointer<UniformLightSampler>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline SampledLight sample(float u) const {
		auto sample = [&](auto ptr)->SampledLight {return ptr->sample(u); };
		return dispatch(sample);
	}

	__both__ inline float pdf(const Light& light) const{
		auto pdf = [&](auto ptr)->float {return ptr->pdf(light); };
		return dispatch(pdf);
	}

};

KRR_NAMESPACE_END