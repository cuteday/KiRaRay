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
	KRR_CALLABLE SampledLight sample(float u)const {
		SampledLight sl = {};
		uint sampleId = u * mLights.size();
		DCHECK(sampleId < mLights.size());
		sl.light = mLights[sampleId];
		sl.pdf = 1.f / mLights.size();
		//DCHECK(0 <= sampleId && sampleId < mLights.size());
		//printf("%lld\n", mLights[0].ptr());

		DCHECK(sl.light.ptr());
		return sl;
	}

	KRR_CALLABLE float pdf(const Light& light)const {
		return 1.f / mLights.size();
	}

	KRR_CALLABLE inter::vector<Light> getLights() {
		return mLights;
	}

private:
	inter::vector<Light> mLights;
};

class LightSampler: TaggedPointer<UniformLightSampler>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE SampledLight sample(float u) const {
		auto sample = [&](auto ptr)->SampledLight {return ptr->sample(u); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(const Light& light) const{
		auto pdf = [&](auto ptr)->float {return ptr->pdf(light); };
		return dispatch(pdf);
	}

};

KRR_NAMESPACE_END