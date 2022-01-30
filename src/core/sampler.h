#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "common.h"

// optix kernels does not support virtual functions...
// considering use tagged pointer like pbrt instead.

KRR_NAMESPACE_BEGIN

using namespace math;

class Sampler {
public:
	using SharedPtr = std::shared_ptr<Sampler>;

	__both__ Sampler() {}

	__both__ virtual void setPixelSample(vec2i samplePixel, uint sampleIndex)
		{ mSamplePixel = samplePixel; }

	__both__ virtual float get1D() = 0;
	__both__ virtual vec2f get2D() = 0;

protected:
	vec2i mSamplePixel = { 0, 0 };
	uint mSampleIndex = 0;
};

class UniformSampler : Sampler {
	__both__ UniformSampler(): Sampler(){}

	__both__ float get1D() override { return 0; }
	__both__ vec2f get2D() override { return vec2f(0); }

};

class LCGSampler {
public:
	using SharedPtr = std::shared_ptr<LCGSampler>;

	__both__ LCGSampler(){}

	__both__ void setSeed(uint seed) { mState = seed; }

	__both__ void setPixel(vec2i samplePixel, uint sampleIndex)  {

		uint v0 = utils::interleave_32bit(vec2ui(samplePixel));
		uint v1 = sampleIndex;

		mState = utils::blockCipherTEA(v0, v1, 16).x;
	}

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
	uint mState;
};

KRR_NAMESPACE_END