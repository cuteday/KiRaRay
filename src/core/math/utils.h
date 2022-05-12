#pragma once

#include "common.h"
#include "math/math.h"
#include "util/check.h"

KRR_NAMESPACE_BEGIN

namespace math{
	namespace utils{
		/*******************************************************
		* colors
		********************************************************/
		KRR_CALLABLE float luminance(vec3f color)
		{
			return dot(color, vec3f(0.299, 0.587, 0.114));
		}

		KRR_CALLABLE float srgb2linear(float sRGBColor)
		{
			if (sRGBColor <= 0.04045)
				return sRGBColor / 12.92;
			else
				return pow((sRGBColor + 0.055) / 1.055, 2.4);
		}

		KRR_CALLABLE vec3f srgb2linear(vec3f sRGBColor)
		{
			return vec3f(srgb2linear(sRGBColor.r), srgb2linear(sRGBColor.g), srgb2linear(sRGBColor.b));
		}

		KRR_CALLABLE float linear2srgb(float linearColor)
		{
			if (linearColor <= 0.0031308)
				return linearColor * 12.92;
			else
				return 1.055 * pow(linearColor, 1.0 / 2.4) - 0.055;
		}

		KRR_CALLABLE vec3f linear2srgb(vec3f linearColor)
		{
			return vec3f(linear2srgb(linearColor.r), linear2srgb(linearColor.g), linear2srgb(linearColor.b));
		}

		/*******************************************************
		* numbers
		********************************************************/
		template<typename T>
		KRR_CALLABLE void extendedGCD(T a, T b, T *x, T *y) {
			if (b == 0) {
				*x = 1;
				*y = 0;
				return;
			}
			T d = a / b, xp, yp;
			extendedGCD(b, a % b, &xp, &yp);
			*x = yp;
			*y = xp - (d * yp);
		}

		template<typename T>
		KRR_CALLABLE T multiplicativeInverse(T a, T n) {
			T x, y;
			extendedGCD(a, n, &x, &y);
			return x % n;
		}

		template<typename T>
		KRR_CALLABLE T lerp(T x, T y, float weight) {
			return (1.f - weight) * x + weight * y;
		}

		/*******************************************************
		* bit tricks
		********************************************************/
	
		KRR_CALLABLE uint interleave_32bit(vec2ui v){
			uint x = v.x & 0x0000ffff;              // x = ---- ---- ---- ---- fedc ba98 7654 3210
			uint y = v.y & 0x0000ffff;

			x = (x | (x << 8)) & 0x00FF00FF;        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
			x = (x | (x << 4)) & 0x0F0F0F0F;        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
			x = (x | (x << 2)) & 0x33333333;        // x = --fe --dc --ba --98 --76 --54 --32 --10
			x = (x | (x << 1)) & 0x55555555;        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

			y = (y | (y << 8)) & 0x00FF00FF;
			y = (y | (y << 4)) & 0x0F0F0F0F;
			y = (y | (y << 2)) & 0x33333333;
			y = (y | (y << 1)) & 0x55555555;

			return x | (y << 1);
		}

		/*******************************************************
		* vectors and coordinates
		********************************************************/

		template <typename T>
		KRR_CALLABLE float lengthSquared(T v) {
			float l = length(v);
			return l * l;
		}

		KRR_CALLABLE float sphericalTriangleArea(vec3f a, vec3f b, vec3f c) {
			return abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)));
		}

		// generate a perpendicular vector which is orthogonal to the given vector
		KRR_CALLABLE vec3f getPerpendicular(const vec3f& u){
			vec3f a = abs(u);
			uint32_t uyx = (a.x - a.y) < 0 ? 1 : 0;
			uint32_t uzx = (a.x - a.z) < 0 ? 1 : 0;
			uint32_t uzy = (a.y - a.z) < 0 ? 1 : 0;
			uint32_t xm = uyx & uzx;
			uint32_t ym = (1 ^ xm) & uzy;
			uint32_t zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
			vec3f v = normalize(cross(u, vec3f(xm, ym, zm)));
			return v;
		}

		// world => y-up
		KRR_CALLABLE vec2f worldToLatLong(const vec3f& dir) {
			vec3f p = normalize(dir);
			vec2f uv;
			uv.x = atan2(p.x, -p.z) / M_2PI + 0.5f;
			uv.y = acos(p.y) * M_1_PI;
			return uv;
		}

		/// <param name="latlong"> in [0, 1]*[0, 1] </param>
		KRR_CALLABLE vec3f latlongToWorld(vec2f latlong)
		{
			float phi = M_PI * (2.f * saturate(latlong.x) - 1.f);
			float theta = M_PI * saturate(latlong.y);
			float sinTheta = sin(theta);
			float cosTheta = cos(theta);
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);
			return { sinTheta * sinPhi, cosTheta, -sinTheta * cosPhi };
		}

		// caetesian, or local frame => z-up 
		KRR_CALLABLE vec2f cartesianToSpherical(const vec3f& v) {
			vec3f nv = normalize(v);
			vec2f sph;
			sph.x = acos(nv.z);
			sph.y = atan2(-nv.y, -nv.x) + M_PI;
			return sph;
		}

		KRR_CALLABLE vec2f cartesianToSphericalNormalized(const vec3f& v) {
			vec2f sph = cartesianToSpherical(v);
			return { sph.x / M_PI, sph.y / M_2PI };
		}

		/// <param name="sph">\phi in [0, 2pi] and \theta in [0, pi]</param>
		KRR_CALLABLE vec3f sphericalToCartesian(float theta, float phi) {
			float sinTheta = sin(theta);
			float cosTheta = cos(theta);
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);
			return { sinTheta * sinPhi, -sinTheta * cosPhi, cosTheta, };
		}

		KRR_CALLABLE vec3f sphericalToCartesian(float sinTheta, float cosTheta, float phi) {
			return vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
		}

		KRR_CALLABLE float sphericalTheta(const vec3f& v) {
			return acos(clamp(v.z, -1.f, 1.f));
		}

		KRR_CALLABLE float sphericalPhi(const vec3f& v) {
			float p = atan2(v.y, v.x);
			return (p < 0) ? (p + 2 * M_PI) : p;
		}

		/*******************************************************
		* hashing utils
		********************************************************/
		KRR_CALLABLE vec2ui blockCipherTEA(uint v0, uint v1, uint iterations = 16)
		{
			uint sum = 0;
			const uint delta = 0x9e3779b9;
			const uint k[4] = { 0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e }; // 128-bit key.
			for (uint i = 0; i < iterations; i++)
			{
				sum += delta;
				v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
				v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
			}
			return vec2ui(v0, v1);
		}

		KRR_CALLABLE uint64_t MixBits(uint64_t v) {
			v ^= (v >> 31);
			v *= 0x7fb5d329728ea185;
			v ^= (v >> 27);
			v *= 0x81dadef4bc2dd44d;
			v ^= (v >> 33);
			return v;
		}

		KRR_CALLABLE uint64_t MurmurHash64A(const unsigned char* key, size_t len,
			uint64_t seed) {
			const uint64_t m = 0xc6a4a7935bd1e995ull;
			const int r = 47;

			uint64_t h = seed ^ (len * m);

			const unsigned char* end = key + 8 * (len / 8);

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

		template <typename... Args>
		KRR_CALLABLE void hashRecursiveCopy(char* buf, Args...);

		template <>
		KRR_CALLABLE void hashRecursiveCopy(char* buf) {}

		template <typename T, typename... Args>
		KRR_CALLABLE void hashRecursiveCopy(char* buf, T v, Args... args) {
			memcpy(buf, &v, sizeof(T));
			hashRecursiveCopy(buf + sizeof(T), args...);
		}

		template <typename... Args>
		KRR_CALLABLE uint64_t Hash(Args... args) {
			// C++, you never cease to amaze: https://stackoverflow.com/a/57246704
			constexpr size_t sz = (sizeof(Args) + ... + 0);
			constexpr size_t n = (sz + 7) / 8;
			uint64_t buf[n];
			hashRecursiveCopy((char*)buf, args...);
			return MurmurHash64A((const unsigned char*)buf, sz, 0);
		}

		template <typename... Args>
		KRR_CALLABLE float HashFloat(Args... args) {
			return uint32_t(Hash(args...)) * 0x1p-32f;
		}

		/*******************************************************
		* bitmask operations
		********************************************************/
		KRR_CALLABLE uint32_t ReverseBits32(uint32_t n) {
#ifdef __CUDA_ARCH__
			return __brev(n);
#else
			n = (n << 16) | (n >> 16);
			n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
			n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
			n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
			n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
			return n;
#endif
		}

		KRR_CALLABLE uint64_t ReverseBits64(uint64_t n) {
#ifdef __CUDA_ARCH__
			return __brevll(n);
#else
			uint64_t n0 = ReverseBits32((uint32_t)n);
			uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
			return (n0 << 32) | n1;
#endif
		}

		// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
		// updated to 64 bits.
		KRR_CALLABLE uint64_t LeftShift2(uint64_t x) {
			x &= 0xffffffff;
			x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
			x = (x ^ (x << 8)) & 0x00ff00ff00ff00ff;
			x = (x ^ (x << 4)) & 0x0f0f0f0f0f0f0f0f;
			x = (x ^ (x << 2)) & 0x3333333333333333;
			x = (x ^ (x << 1)) & 0x5555555555555555;
			return x;
		}

		KRR_CALLABLE uint64_t EncodeMorton2(uint32_t x, uint32_t y) {
			return (LeftShift2(y) << 1) | LeftShift2(x);
		}

		KRR_CALLABLE uint32_t LeftShift3(uint32_t x) {
			DCHECK_LE(x, (1u << 10));
			if (x == (1 << 10))
				--x;
			x = (x | (x << 16)) & 0b00000011000000000000000011111111;
			// x = ---- --98 ---- ---- ---- ---- 7654 3210
			x = (x | (x << 8)) & 0b00000011000000001111000000001111;
			// x = ---- --98 ---- ---- 7654 ---- ---- 3210
			x = (x | (x << 4)) & 0b00000011000011000011000011000011;
			// x = ---- --98 ---- 76-- --54 ---- 32-- --10
			x = (x | (x << 2)) & 0b00001001001001001001001001001001;
			// x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
			return x;
		}

		KRR_CALLABLE uint32_t EncodeMorton3(float x, float y, float z) {
			DCHECK_GE(x, 0);
			DCHECK_GE(y, 0);
			DCHECK_GE(z, 0);
			return (LeftShift3(z) << 2) | (LeftShift3(y) << 1) | LeftShift3(x);
		}

		KRR_CALLABLE uint32_t Compact1By1(uint64_t x) {
			// TODO: as of Haswell, the PEXT instruction could do all this in a
			// single instruction.
			// x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
			x &= 0x5555555555555555;
			// x = --fe --dc --ba --98 --76 --54 --32 --10
			x = (x ^ (x >> 1)) & 0x3333333333333333;
			// x = ---- fedc ---- ba98 ---- 7654 ---- 3210
			x = (x ^ (x >> 2)) & 0x0f0f0f0f0f0f0f0f;
			// x = ---- ---- fedc ba98 ---- ---- 7654 3210
			x = (x ^ (x >> 4)) & 0x00ff00ff00ff00ff;
			// x = ---- ---- ---- ---- fedc ba98 7654 3210
			x = (x ^ (x >> 8)) & 0x0000ffff0000ffff;
			// ...
			x = (x ^ (x >> 16)) & 0xffffffff;
			return x;
		}

		KRR_CALLABLE void DecodeMorton2(uint64_t v, uint32_t* x, uint32_t* y) {
			*x = Compact1By1(v);
			*y = Compact1By1(v >> 1);
		}

		KRR_CALLABLE uint32_t Compact1By2(uint32_t x) {
			x &= 0x09249249;                   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
			x = (x ^ (x >> 2)) & 0x030c30c3;   // x = ---- --98 ---- 76-- --54 ---- 32-- --10
			x = (x ^ (x >> 4)) & 0x0300f00f;   // x = ---- --98 ---- ---- 7654 ---- ---- 3210
			x = (x ^ (x >> 8)) & 0xff0000ff;   // x = ---- --98 ---- ---- ---- ---- 7654 3210
			x = (x ^ (x >> 16)) & 0x000003ff;  // x = ---- ---- ---- ---- ---- --98 7654 3210
			return x;
		}
	}
}

KRR_NAMESPACE_END