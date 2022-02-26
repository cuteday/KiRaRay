#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "common.h"
#include "taggedptr.h"

// optix kernels does not support virtual functions...
// considering use tagged pointer like pbrt instead.

KRR_NAMESPACE_BEGIN

using namespace math;

class LCGSampler {
public:
	using SharedPtr = std::shared_ptr<LCGSampler>;

	LCGSampler() = default;

	__both__ void setSeed(uint seed) { mState = seed; }

	__both__ void setPixelSample(vec2i samplePixel, uint sampleIndex)  {
		uint v0 = utils::interleave_32bit(vec2ui(samplePixel));
		uint v1 = sampleIndex;
		mState = utils::blockCipherTEA(v0, v1, 16).x;
	}

	// return u in [0, 1)
	__both__ float get1D() {
		const uint LCG_A = 1664525u;
		const uint LCG_C = 1013904223u;
		mState = (LCG_A * mState + LCG_C);
		return (mState & 0x00FFFFFF) / (float)0x01000000;
	}

	__both__ vec2f get2D() {
		return { get1D(), get1D() };
	}

private:
	uint mState = 0;
};



class Sampler :public TaggedPointer<LCGSampler>{
public:
	using SharedPtr = std::shared_ptr<Sampler>;
	using TaggedPointer::TaggedPointer;

	__both__ inline void setPixelSample(vec2i samplePixel, uint sampleIndex){
		auto setPixelSample = [&](auto ptr) ->void {return ptr->setPixelSample(samplePixel, sampleIndex); };
		return dispatch(setPixelSample);
	}

	__both__ inline float get1D() {
		auto get1D = [&](auto ptr) ->float {return ptr->get1D(); };
		return dispatch(get1D);
	};
	__both__ inline vec2f get2D() {
		auto get2D = [&](auto ptr) ->vec2f {return ptr->get2D(); };
		return dispatch(get2D);
	};

protected:
	vec2i mSamplePixel = { 0, 0 };
	uint mSampleIndex = 0;
};

KRR_NAMESPACE_END