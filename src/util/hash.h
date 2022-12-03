/* float number hashing from pbrt-v4 */
#pragma once
#include "common.h"


KRR_NAMESPACE_BEGIN

/*******************************************************
 * bit tricks
 ********************************************************/

KRR_CALLABLE uint interleave_32bit(Vector2ui v) {
	uint x = v[0] & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	uint y = v[1] & 0x0000ffff;

	x = (x | (x << 8)) & 0x00FF00FF; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x | (x << 4)) & 0x0F0F0F0F; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x | (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x | (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

	y = (y | (y << 8)) & 0x00FF00FF;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	y = (y | (y << 2)) & 0x33333333;
	y = (y | (y << 1)) & 0x55555555;

	return x | (y << 1);
}

/*******************************************************
 * hashing utils
 ********************************************************/
KRR_CALLABLE Vector2ui blockCipherTEA(uint v0, uint v1, uint iterations = 16) {
	uint sum		 = 0;
	const uint delta = 0x9e3779b9;
	const uint k[4]	 = { 0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e }; // 128-bit key.
	for (uint i = 0; i < iterations; i++) {
		sum += delta;
		v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
		v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
	}
	return Vector2ui(v0, v1);
}

// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
KRR_CALLABLE uint64_t MurmurHash64A(const unsigned char *key, size_t len, uint64_t seed) {
	const uint64_t m = 0xc6a4a7935bd1e995ull;
	const int r		 = 47;

	uint64_t h = seed ^ (len * m);

	const unsigned char *end = key + 8 * (len / 8);

	while (key != end) {
		uint64_t k;
		std::memcpy(&k, key, sizeof(uint64_t));
		key += 8;

		k *= m;
		k ^= k >> r;
		k *= m;

		h ^= k;
		h *= m;
	}

	switch (len & 7) {
		case 7:
			h ^= uint64_t(key[6]) << 48;
		case 6:
			h ^= uint64_t(key[5]) << 40;
		case 5:
			h ^= uint64_t(key[4]) << 32;
		case 4:
			h ^= uint64_t(key[3]) << 24;
		case 3:
			h ^= uint64_t(key[2]) << 16;
		case 2:
			h ^= uint64_t(key[1]) << 8;
		case 1:
			h ^= uint64_t(key[0]);
			h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}

// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
KRR_CALLABLE uint64_t MixBits(uint64_t v);

uint64_t MixBits(uint64_t v) {
	v ^= (v >> 31);
	v *= 0x7fb5d329728ea185;
	v ^= (v >> 27);
	v *= 0x81dadef4bc2dd44d;
	v ^= (v >> 33);
	return v;
}

template <typename T>
KRR_CALLABLE uint64_t HashBuffer(const T *ptr, size_t size, uint64_t seed = 0) {
	return MurmurHash64A((const unsigned char *) ptr, size, seed);
}

template <typename... Args> KRR_CALLABLE uint64_t Hash(Args... args);

template <typename... Args> KRR_CALLABLE void hashRecursiveCopy(char *buf, Args...);

template <> KRR_CALLABLE void hashRecursiveCopy(char *buf) {}

template <typename T, typename... Args>
KRR_CALLABLE void hashRecursiveCopy(char *buf, T v, Args... args) {
	memcpy(buf, &v, sizeof(T));
	hashRecursiveCopy(buf + sizeof(T), args...);
}

template <typename... Args> KRR_CALLABLE uint64_t Hash(Args... args) {
	// C++, you never cease to amaze: https://stackoverflow.com/a/57246704
	constexpr size_t sz = (sizeof(Args) + ... + 0);
	constexpr size_t n	= (sz + 7) / 8;
	uint64_t buf[n];
	hashRecursiveCopy((char *) buf, args...);
	return MurmurHash64A((const unsigned char *) buf, sz, 0);
}

template <typename... Args> KRR_CALLABLE float HashFloat(Args... args) {
	return uint32_t(Hash(args...)) * 0x1p-32f;
}


KRR_NAMESPACE_END