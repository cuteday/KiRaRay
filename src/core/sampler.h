#pragma once
#include "common.h"
#include "math/math.h"
#include "math/utils.h"
#include "taggedptr.h"
#include "util/lowdiscrepancy.h"

KRR_NAMESPACE_BEGIN

using namespace math;

class PCGSampler {
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL
public:
	PCGSampler() = default;

	KRR_CALLABLE void initialize(){};

	KRR_CALLABLE void setSeed(uint64_t initstate, uint64_t initseq = 1) {
		state = 0U;
		inc	  = (initseq << 1u) | 1u;
		nextUint();
		state += initstate;
		nextUint();
	}

	KRR_CALLABLE void setPixelSample(Vector2ui samplePixel, uint sampleIndex) {
		uint s0 = utils::interleave_32bit(samplePixel);
		uint s1 = sampleIndex;
		setSeed(s0, s1);
	}

	// return u in [0, 1)
	KRR_CALLABLE float get1D() { return nextFloat(); }

	// return an independent 2D sampled vector in [0, 1)^2
	KRR_CALLABLE Vector2f get2D() { return { get1D(), get1D() }; }

	KRR_CALLABLE void advance(int64_t delta = (1ll < 32)) {
		uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u, acc_plus = 0u;

		while (delta > 0) {
			if (delta & 1) {
				acc_mult *= cur_mult;
				acc_plus = acc_plus * cur_mult + cur_plus;
			}
			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta /= 2;
		}
		state = acc_mult * state + acc_plus;
	}

private:
	KRR_CALLABLE uint32_t nextUint() {
		uint64_t oldstate	= state;
		state				= oldstate * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot		= (uint32_t) (oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	}

	KRR_CALLABLE double nextDouble() {
		/* Trick from MTGP: generate an uniformly distributed
			double precision number in [1,2) and subtract 1. */
		union {
			uint64_t u;
			double d;
		} x;
		x.u = ((uint64_t) nextUint() << 20) | 0x3ff0000000000000ULL;
		return x.d - 1.0;
	}

	KRR_CALLABLE float nextFloat() {
		/* Trick from MTGP: generate an uniformly distributed
			single precision number in [1,2) and subtract 1. */
		union {
			uint32_t u;
			float f;
		} x;
		x.u = (nextUint() >> 9) | 0x3f800000u;
		return x.f - 1.0f;
	}

	uint64_t state; // RNG state.  All values are possible.
	uint64_t inc;	// Controls which RNG sequence (stream) is selected. Must
					// *always* be odd.
};


class LCGSampler {
public:
	LCGSampler() = default;

	KRR_CALLABLE void initialize(){};

	KRR_CALLABLE void setSeed(uint seed) { state = seed; }

	KRR_CALLABLE void setPixelSample(Vector2ui samplePixel, uint sampleIndex) {
		uint v0 = utils::interleave_32bit(Vector2ui(samplePixel));
		uint v1 = sampleIndex;
		state	= utils::blockCipherTEA(v0, v1, 16)[0];
	}

	// return u in [0, 1)
	KRR_CALLABLE float get1D() {
		const uint LCG_A = 1664525u;
		const uint LCG_C = 1013904223u;
		state			 = (LCG_A * state + LCG_C);
		return (state & 0x00FFFFFF) / (float) 0x01000000;
	}

	KRR_CALLABLE Vector2f get2D() { return { get1D(), get1D() }; }

private:
	uint state = 0;
};


class HaltonSampler {
public:
	HaltonSampler() = default;

	KRR_CALLABLE void initialize(RandomizeStrategy randomize = RandomizeStrategy::None) {
		this->randomize = randomize;
		for (int i = 0; i < 2; ++i) {
			int base  = (i == 0) ? 2 : 3;
			int scale = 1, exp = 0;
			while (scale < maxHaltonResolution) {
				scale *= base;
				++exp;
			}
			baseScales[i]	 = scale;
			baseExponents[i] = exp;
		}
		// Compute multiplicative inverses for _baseScales_
		multInverse[0] = multiplicativeInverse(baseScales[1], baseScales[0]);
		multInverse[1] = multiplicativeInverse(baseScales[0], baseScales[1]);
	}

	KRR_CALLABLE RandomizeStrategy getRandomizeStrategy() const { return randomize; }

	KRR_CALLABLE void setPixelSample(Vector2ui p, int sampleIndex) { return setPixelSample(p, sampleIndex, 0); }

	KRR_CALLABLE void setPixelSample(Vector2ui p, int sampleIndex, int dim) {
		haltonIndex		 = 0;
		int sampleStride = baseScales[0] * baseScales[1];
		// Compute Halton sample index for first sample in pixel _p_
		if (sampleStride > 1) {
			Vector2ui pm(p[0] % maxHaltonResolution, p[1] % maxHaltonResolution);
			for (int i = 0; i < 2; ++i) {
				uint64_t dimOffset = (i == 0) ? InverseRadicalInverse(pm[i], 2, baseExponents[i])
											  : InverseRadicalInverse(pm[i], 3, baseExponents[i]);
				haltonIndex += dimOffset * (sampleStride / baseScales[i]) * multInverse[i];
			}
			haltonIndex %= sampleStride;
		}
		haltonIndex += sampleIndex * sampleStride;
		dimension = max(2, dim);
	}

	KRR_CALLABLE float get1D() {
		if (dimension >= PrimeTableSize)
			dimension = 0;
		return sampleDimension(dimension++);
	}

	KRR_CALLABLE Vector2f get2D() {
		if (dimension + 1 >= PrimeTableSize)
			dimension = 0;
		int dim = dimension;
		dimension += 2;
		return { sampleDimension(dim), sampleDimension(dim + 1) };
	}

private:
	KRR_CALLABLE static uint64_t multiplicativeInverse(int64_t a, int64_t n) {
		int64_t x, y;
		utils::extendedGCD(a, n, &x, &y);
		return mod(x, n);
	}

	KRR_CALLABLE float sampleDimension(int dimension) const {
		if (randomize == RandomizeStrategy::None)
			return RadicalInverse(dimension, haltonIndex);
		else {
			DCHECK_EQ(randomize, RandomizeStrategy::Owen);
			return OwenScrambledRadicalInverse(dimension, haltonIndex, MixBits(1 + (dimension << 4)));
		}
	}

	RandomizeStrategy randomize;
	static constexpr int maxHaltonResolution = 128;
	Vector2i baseScales, baseExponents;
	int multInverse[2];
	int64_t haltonIndex = 0;
	int dimension		= 0;
};


class Sampler : public TaggedPointer<PCGSampler, LCGSampler, HaltonSampler> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE void setPixelSample(Vector2ui samplePixel, uint sampleIndex) {
		auto setPixelSample = [&](auto ptr) -> void { return ptr->setPixelSample(samplePixel, sampleIndex); };
		return dispatch(setPixelSample);
	}

	KRR_CALLABLE float get1D() {
		auto get1D = [&](auto ptr) -> float { return ptr->get1D(); };
		return dispatch(get1D);
	};
	KRR_CALLABLE Vector2f get2D() {
		auto get2D = [&](auto ptr) -> Vector2f { return ptr->get2D(); };
		return dispatch(get2D);
	};
};

KRR_NAMESPACE_END