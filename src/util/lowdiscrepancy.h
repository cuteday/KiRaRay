// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#pragma once
#include <algorithm>
#include <memory>
#include <string>

#include "common.h"
#include "interop.h"
#include "check.h"
#include "tables.h"
#include "math/math.h"
#include "math/utils.h"

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace math::utils;

KRR_CALLABLE int PermutationElement(uint32_t i, uint32_t l, uint32_t p) {
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

// DigitPermutation Definition
class DigitPermutation {
public:
    // DigitPermutation Public Methods
    DigitPermutation() = default;
    DigitPermutation(int base, uint32_t seed, Allocator alloc) : base(base) {
        CHECK_LT(base, 65536);  // uint16_t
        // Compute number of digits needed for _base_
        nDigits = 0;
        float invBase = (float)1 / (float)base, invBaseM = 1;
        while (1 - (base - 1) * invBaseM < 1) {
            ++nDigits;
            invBaseM *= invBase;
        }

        permutations = alloc.allocate_object<uint16_t>(nDigits * base);
        // Compute random permutations for all digits
        for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
            uint64_t dseed = Hash(base, digitIndex, seed);
            for (int digitValue = 0; digitValue < base; ++digitValue) {
                int index = digitIndex * base + digitValue;
                permutations[index] = PermutationElement(digitValue, base, dseed);
            }
        }
    }

    KRR_CALLABLE
        int Permute(int digitIndex, int digitValue) const {
        DCHECK_LT(digitIndex, nDigits);
        DCHECK_LT(digitValue, base);
        return permutations[digitIndex * base + digitValue];
    }

private:
    // DigitPermutation Private Members
    int base, nDigits;
    uint16_t* permutations;
};

//inline inter::vector<DigitPermutation>* ComputeRadicalInversePermutations(uint32_t seed,
//    Allocator alloc = {}) {
//    inter::vector<DigitPermutation>* perms =
//        alloc.new_object<inter::vector<DigitPermutation>>(alloc);
//    perms->resize(PrimeTableSize);
//    for (int i = 0; i < PrimeTableSize; ++i)
//        (*perms)[i] = DigitPermutation(Primes[i], seed, alloc);
//    return perms;
//}

// NoRandomizer Definition
struct NoRandomizer {
    KRR_CALLABLE
        uint32_t operator()(uint32_t v) const { return v; }
};

// Low Discrepancy Inline Functions
KRR_CALLABLE float RadicalInverse(int baseIndex, uint64_t a) {
    int base = Primes[baseIndex];
    float invBase = (float)1 / (float)base, invBaseN = 1;
    uint64_t reversedDigits = 0;
    while (a) {
        // Extract least significant digit from _a_ and update _reversedDigits_
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return min(reversedDigits * invBaseN, OneMinusEpsilon);
}

KRR_CALLABLE uint64_t InverseRadicalInverse(uint64_t inverse, int base,
    int nDigits) {
    uint64_t index = 0;
    for (int i = 0; i < nDigits; ++i) {
        uint64_t digit = inverse % base;
        inverse /= base;
        index = index * base + digit;
    }
    return index;
}

KRR_CALLABLE float ScrambledRadicalInverse(int baseIndex, uint64_t a,
    const DigitPermutation& perm) {
    int base = Primes[baseIndex];
    float invBase = (float)1 / (float)base, invBaseM = 1;
    uint64_t reversedDigits = 0;
    int digitIndex = 0;
    while (1 - (base - 1) * invBaseM < 1) {
        // Permute least significant digit from _a_ and update _reversedDigits_
        uint64_t next = a / base;
        int digitValue = a - next * base;
        reversedDigits = reversedDigits * base + perm.Permute(digitIndex, digitValue);
        invBaseM *= invBase;
        ++digitIndex;
        a = next;
    }
    return min(invBaseM * reversedDigits, OneMinusEpsilon);
}

KRR_CALLABLE float OwenScrambledRadicalInverse(int baseIndex, uint64_t a,
    uint32_t hash) {
    int base = Primes[baseIndex];
    float invBase = (float)1 / (float)base, invBaseM = 1;
    uint64_t reversedDigits = 0;
    int digitIndex = 0;
    while (1 - invBaseM < 1) {
        // Compute Owen-scrambled digit for _digitIndex_
        uint64_t next = a / base;
        int digitValue = a - next * base;
        uint32_t digitHash = MixBits(hash ^ reversedDigits);
        digitValue = PermutationElement(digitValue, base, digitHash);
        reversedDigits = reversedDigits * base + digitValue;
        invBaseM *= invBase;
        ++digitIndex;
        a = next;
    }
    return min(invBaseM * reversedDigits, OneMinusEpsilon);
}

KRR_CALLABLE uint32_t MultiplyGenerator(inter::span<const uint32_t> C, uint32_t a) {
    uint32_t v = 0;
    for (int i = 0; a != 0; ++i, a >>= 1)
        if (a & 1)
            v ^= C[i];
    return v;
}

KRR_CALLABLE float BlueNoiseSample(vec2i p, int instance) {
    auto HashPerm = [&](uint64_t index) -> int {
        return uint32_t(MixBits(index ^ (0x55555555 * instance)) >> 24) % 24;
    };

    int nBase4Digits = 8;  // Log2Int(256)
    p.x &= 255;
    p.y &= 255;
    uint64_t mortonIndex = EncodeMorton2(p.x, p.y);

    static const uint8_t permutations[24][4] = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 2, 1},
        {0, 3, 1, 2}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0},
        {1, 3, 2, 0}, {1, 3, 0, 2}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3},
        {2, 0, 3, 1}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 1, 2, 0}, {3, 1, 0, 2},
        {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2} };

    uint32_t sampleIndex = 0;
    for (int i = nBase4Digits - 1; i >= 0; --i) {
        int digitShift = 2 * i;
        int digit = (mortonIndex >> digitShift) & 3;
        int p = HashPerm(mortonIndex >> (digitShift + 2));
        digit = permutations[p][digit];
        sampleIndex |= digit << digitShift;
    }

    return ReverseBits32(sampleIndex) * 0x1p-32f;
}

// BinaryPermuteScrambler Definition
struct BinaryPermuteScrambler {
    KRR_CALLABLE
        BinaryPermuteScrambler(uint32_t perm) : permutation(perm) {}
    KRR_CALLABLE
        uint32_t operator()(uint32_t v) const { return permutation ^ v; }
    uint32_t permutation;
};

// FastOwenScrambler Definition
struct FastOwenScrambler {
    KRR_CALLABLE
        FastOwenScrambler(uint32_t seed) : seed(seed) {}
    // FastOwenScrambler Public Methods
    KRR_CALLABLE
        uint32_t operator()(uint32_t v) const {
        v = ReverseBits32(v);
        v ^= v * 0x3d20adea;
        v += seed;
        v *= (seed >> 16) | 1;
        v ^= v * 0x05526c56;
        v ^= v * 0x53a22864;
        return ReverseBits32(v);
    }

    uint32_t seed;
};

// OwenScrambler Definition
struct OwenScrambler {
    KRR_CALLABLE
        OwenScrambler(uint32_t seed) : seed(seed) {}
    // OwenScrambler Public Methods
    KRR_CALLABLE
        uint32_t operator()(uint32_t v) const {
        if (seed & 1)
            v ^= 1u << 31;
        for (int b = 1; b < 32; ++b) {
            // Apply Owen scrambling to binary digit _b_ in _v_
            uint32_t mask = (~0u) << (32 - b);
            if ((uint32_t)MixBits((v & mask) ^ seed) & (1u << b))
                v ^= 1u << (31 - b);
        }
        return v;
    }

    uint32_t seed;
};

// RandomizeStrategy Definition
enum class RandomizeStrategy { None, PermuteDigits, FastOwen, Owen };

//template <typename R>
//KRR_CALLABLE float SobolSample(int64_t a, int dimension, R randomizer) {
//    DCHECK_LT(dimension, NSobolDimensions);
//    DCHECK(a >= 0 && a < (1ull << SobolMatrixSize));
//    // Compute initial Sobol sample _v_ using generator matrices
//    uint32_t v = 0;
//    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
//        if (a & 1)
//            v ^= SobolMatrices32[i];
//
//    // Randomize Sobol sample and return floating-point value
//    v = randomizer(v);
//    return std::min(v * 0x1p-32f, floatOneMinusEpsilon);
//}

//KRR_CALLABLE
//    uint64_t SobolIntervalToIndex(uint32_t m, uint64_t frame, vec2i p) {
//    if (m == 0)
//        return frame;
//
//    const uint32_t m2 = m << 1;
//    uint64_t index = uint64_t(frame) << m2;
//
//    uint64_t delta = 0;
//    for (int c = 0; frame; frame >>= 1, ++c)
//        if (frame & 1)  // Add flipped column m + c + 1.
//            delta ^= VdCSobolMatrices[m - 1][c];
//
//    // flipped b
//    uint64_t b = (((uint64_t)((uint32_t)p.x) << m) | ((uint32_t)p.y)) ^ delta;
//
//    for (int c = 0; b; b >>= 1, ++c)
//        if (b & 1)  // Add column 2 * m - c.
//            index ^= VdCSobolMatricesInv[m - 1][c];
//
//    return index;
//}

KRR_NAMESPACE_END
