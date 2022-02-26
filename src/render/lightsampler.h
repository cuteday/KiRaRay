#pragma once

#include "common.h"
#include "taggedptr.h"
#include "light.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

struct SampledLight{
	Light light;
	float pdf;
};

class UniformLightSampler{
public:
	UniformLightSampler(TypedBuffer<Light> lights):mLights(lights) {}

	// assumes u in [0, 1)
	__both__ inline SampledLight sample(float u){
		SampledLight sl = {};
		uint sampleId = u * mLights.size();
		sl.light = mLights[sampleId];
		sl.pdf = 1 / mLights.size();
		return sl;
	}

	__both__ inline float pdf() {
		return 1 / mLights.size();
	}

private:
	TypedBuffer<Light> mLights;
};

class LightSampler: TaggedPointer<UniformLightSampler>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline SampledLight sample(float u){
		auto sample = [&](auto ptr)->SampledLight {return ptr->sample(); };
		return dispatch(sample);
	}

	__both__ inline float pdf(){
		auto pdf = [&](auto ptr)->float {return ptr->pdf(); };
		return dispatch(pdf);
	}

};

KRR_NAMESPACE_END