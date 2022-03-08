#pragma once

#include "math/math.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

namespace math{

	namespace utils{
		/*******************************************************
		* colors
		********************************************************/
		__both__ inline float luminance(vec3f color)
		{
			return dot(color, vec3f(0.299, 0.587, 0.114));
		}

		__both__ inline float srgb2linear(float sRGBColor)
		{
			if (sRGBColor <= 0.04045)
				return sRGBColor / 12.92;
			else
				return pow((sRGBColor + 0.055) / 1.055, 2.4);
		}

		__both__ inline vec3f srgb2linear(vec3f sRGBColor)
		{
			return vec3f(srgb2linear(sRGBColor.r), srgb2linear(sRGBColor.g), srgb2linear(sRGBColor.b));
		}

		__both__ inline float linear2srgb(float linearColor)
		{
			if (linearColor <= 0.0031308)
				return linearColor * 12.92;
			else
				return 1.055 * pow(linearColor, 1.0 / 2.4) - 0.055;
		}

		__both__ inline vec3f linear2srgb(vec3f linearColor)
		{
			return vec3f(linear2srgb(linearColor.r), linear2srgb(linearColor.g), linear2srgb(linearColor.b));
		}

		/*******************************************************
		* numbers
		********************************************************/
		template<typename T>
		__both__ inline void extendedGCD(T a, T b, T *x, T *y) {
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
		__both__ inline T multiplicativeInverse(T a, T n) {
			T x, y;
			extendedGCD(a, n, &x, &y);
			return x % n;
		}

		template<typename T>
		__both__ inline T lerp(T x, T y, float weight) {
			return (1.f - weight) * x + weight * y;
		}

		/*******************************************************
		* bit tricks
		********************************************************/
	
		__both__ inline uint interleave_32bit(vec2ui v){
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
		__both__ inline float lengthSquared(T v) {
			float l = length(v);
			return l * l;
		}

		__both__ inline float sphericalTriangleArea(vec3f a, vec3f b, vec3f c) {
			return abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)));
		}

		// generate a perpendicular vector which is orthogonal to the given vector
		__both__ inline vec3f getPerpendicular(const vec3f& u){
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
		__both__ inline vec2f worldToLatLong(const vec3f& dir) {
			vec3f p = normalize(dir);
			vec2f uv;
			uv.x = atan2(p.x, -p.z) / M_2PI + 0.5f;
			uv.y = acos(p.y) * M_1_PI;
			return uv;
		}

		/// <param name="latlong"> in [0, 1]*[0, 1] </param>
		__both__ inline vec3f latlongToWorld(vec2f latlong)
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
		__both__ inline vec2f cartesianToSpherical(const vec3f& v) {
			vec3f nv = normalize(v);
			vec2f sph;
			sph.x = acos(nv.z);
			sph.y = atan2(-nv.y, -nv.x) + M_PI;
			return sph;
		}

		__both__ inline vec2f cartesianToSphericalNormalized(const vec3f& v) {
			vec2f sph = cartesianToSpherical(v);
			return { sph.x / M_PI, sph.y / M_2PI };
		}

		/// <param name="sph">\phi in [0, 2pi] and \theta in [0, pi]</param>
		__both__ inline vec3f sphericalToCartesian(float theta, float phi) {
			float sinTheta = sin(theta);
			float cosTheta = cos(theta);
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);
			return { sinTheta * sinPhi, -sinTheta * cosPhi, cosTheta, };
		}

		__both__ inline vec3f sphericalToCartesian(float sinTheta, float cosTheta, float phi) {
			return vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
		}

		__both__ inline float sphericalTheta(const vec3f& v) {
			return acos(clamp(v.z, -1.f, 1.f));
		}

		__both__ inline float sphericalPhi(const vec3f& v) {
			float p = atan2(v.y, v.x);
			return (p < 0) ? (p + 2 * M_PI) : p;
		}

		__both__ inline vec3f offsetRayOrigin(vec3f p, vec3f n, vec3f w) {
			vec3f offset = n * 1e-6f;
			if (dot(n, w) < 0.f)
				offset = -offset;
			return p + offset;
		}

		/*******************************************************
		* hashing utils
		********************************************************/
		__both__ inline vec2ui blockCipherTEA(uint v0, uint v1, uint iterations = 16)
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

		/*******************************************************
		* low discrepancy
		********************************************************/
		//__both__ inline float RadicalInverse(int baseIndex, uint64_t a) {
		//	int base = Primes[baseIndex];
		//	float invBase = (float)1 / (float)base, invBaseN = 1;
		//	uint64_t reversedDigits = 0;
		//	while (a) {
		//		// Extract least significant digit from _a_ and update _reversedDigits_
		//		uint64_t next = a / base;
		//		uint64_t digit = a - next * base;
		//		reversedDigits = reversedDigits * base + digit;
		//		invBaseN *= invBase;
		//		a = next;
		//	}
		//	return min(reversedDigits * invBaseN, OneMinusEpsilon);
		//}
	}
}

KRR_NAMESPACE_END