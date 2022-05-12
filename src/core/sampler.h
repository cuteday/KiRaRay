#pragma once
#include "math/math.h"
#include "math/utils.h"
#include "util/lowdiscrepancy.h"
#include "common.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

using namespace math;

class LCGSampler {
public:
	LCGSampler() = default;

	__both__ void initialize() {};

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

class HaltonSampler {
public:
	HaltonSampler() = default;

	__both__ void initialize(RandomizeStrategy randomize = RandomizeStrategy::None) {
		this->randomize = randomize;
		for (int i = 0; i < 2; ++i) {
			int base = (i == 0) ? 2 : 3;
			int scale = 1, exp = 0;
			while (scale < maxHaltonResolution) {
				scale *= base;
				++exp;
			}
			baseScales[i] = scale;
			baseExponents[i] = exp;
		}
		// Compute multiplicative inverses for _baseScales_
		multInverse[0] = multiplicativeInverse(baseScales[1], baseScales[0]);
		multInverse[1] = multiplicativeInverse(baseScales[0], baseScales[1]);
	}

	__both__ RandomizeStrategy getRandomizeStrategy() const { return randomize; }

	__both__ void setPixelSample(vec2i p, int sampleIndex) {
		return setPixelSample(p, sampleIndex, 0);
	}

	__both__ void setPixelSample(vec2i p, int sampleIndex, int dim) {
		haltonIndex = 0;
		int sampleStride = baseScales[0] * baseScales[1];
		// Compute Halton sample index for first sample in pixel _p_
		if (sampleStride > 1) {
			vec2i pm(p[0] % maxHaltonResolution, p[1] % maxHaltonResolution);
			for (int i = 0; i < 2; ++i) {
				uint64_t dimOffset =
					(i == 0) ? InverseRadicalInverse(pm[i], 2, baseExponents[i])
					: InverseRadicalInverse(pm[i], 3, baseExponents[i]);
				haltonIndex +=
					dimOffset * (sampleStride / baseScales[i]) * multInverse[i];
			}
			haltonIndex %= sampleStride;
		}
		haltonIndex += sampleIndex * sampleStride;
		dimension = max(2, dim);
	}

	__both__ float get1D() {
		if (dimension >= PrimeTableSize)
			dimension = 0;
		return sampleDimension(dimension++);
	}

	__both__ vec2f get2D() {
		if (dimension + 1 >= PrimeTableSize)
			dimension = 0;
		int dim = dimension;
		dimension += 2;
		return { sampleDimension(dim), sampleDimension(dim + 1) };
	}

private:
	// HaltonSampler Private Methods
	__both__ static uint64_t multiplicativeInverse(int64_t a, int64_t n) {
		int64_t x, y;
		utils::extendedGCD(a, n, &x, &y);
		return mod(x, n);
	}

	__both__ float sampleDimension(int dimension) const {
		DCHECK_LE(dimension, maxHaltonResolution);
		DCHECK_LE(maxHaltonResolution, PrimeTableSize);
		if (randomize == RandomizeStrategy::None)
			return RadicalInverse(dimension, haltonIndex);
		else {
			DCHECK_EQ(randomize, RandomizeStrategy::Owen);
			return OwenScrambledRadicalInverse(dimension, haltonIndex,
				MixBits(1 + (dimension << 4)));
		}
	}

	RandomizeStrategy randomize;
	static constexpr int maxHaltonResolution = 256;
	vec2i baseScales, baseExponents;
	int multInverse[2];
	int64_t haltonIndex = 0;
	int dimension = 0;
};

class Sampler :public TaggedPointer<LCGSampler, HaltonSampler>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE void setPixelSample(vec2i samplePixel, uint sampleIndex){
		auto setPixelSample = [&](auto ptr) ->void {return ptr->setPixelSample(samplePixel, sampleIndex); };
		return dispatch(setPixelSample);
	}

	KRR_CALLABLE float get1D() {
		auto get1D = [&](auto ptr) ->float {return ptr->get1D(); };
		return dispatch(get1D);
	};
	KRR_CALLABLE vec2f get2D() {
		auto get2D = [&](auto ptr) ->vec2f {return ptr->get2D(); };
		return dispatch(get2D);
	};

protected:
	vec2i mSamplePixel = { 0, 0 };
	uint mSampleIndex = 0;
};

KRR_NAMESPACE_END