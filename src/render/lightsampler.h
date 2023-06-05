#pragma once

#include "common.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "light.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

struct SampledLight {
	Light light;
	float pdf;
};

class UniformLightSampler {
public:
	UniformLightSampler() = default;

	UniformLightSampler(const TypedBufferView<Light>& lights) : mLights(lights) {}

	// assumes u in [0, 1)
	KRR_CALLABLE SampledLight sample(float u) const {
		SampledLight sl = {};
		uint sampleId	= u * mLights.size();
		DCHECK_LT(sampleId, mLights.size());
		sl.light = mLights[sampleId];
		sl.pdf	 = 1.f / mLights.size();
		return sl;
	}

	KRR_CALLABLE float pdf(const Light &light) const { return 1.f / mLights.size(); }

	KRR_CALLABLE TypedBufferView<Light> getLights() { return mLights; }

private:
	TypedBufferView<Light> mLights;
};

class LightSampler : public TaggedPointer<UniformLightSampler> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE SampledLight sample(float u) const {
		auto sample = [&](auto ptr) -> SampledLight { return ptr->sample(u); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(const Light &light) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdf(light); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END