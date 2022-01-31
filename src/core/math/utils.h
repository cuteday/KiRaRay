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
		* vectors
		********************************************************/

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

		/*******************************************************
		* sampling distributions
		********************************************************/

		__both__ inline vec3f uniformSampleHemisphere(const vec2f& u) {
			float z = u[0];
			float r = sqrt(max(0.f, (float)1.f - z * z));
			float phi = 2 * M_PI * u[1];
			return vec3f(r * cos(phi), r * sin(phi), z);
		}

		__both__ inline vec2f uniformSampleDisk(const vec2f& u) {
			// simpler method derived using marginal distribution...
			//float r = sqrt(u[0]);
			//float theta = 2 * M_PI * u[1];
			//return vec2f(r * cos(theta), r * sin(theta));
				
			// the concentric method
			vec2f uOffset = 2.f * u - vec2f(1, 1);
			if (uOffset.x == 0 && uOffset.y == 0)
				return vec2f(0, 0);
			float theta, r;
			if (fabs(uOffset.x) > fabs(uOffset.y)) {
				r = uOffset.x;
				theta = M_PI / 4 * (uOffset.y / uOffset.x);
			}
			else {
				r = uOffset.y;
				theta = M_PI / 2 - M_PI / 4 * (uOffset.x / uOffset.y);
			}
			return r * vec2f(cos(theta), sin(theta));
		}

		__both__ inline vec3f cosineSampleHemisphere(const vec2f& u) {
			vec2f d = uniformSampleDisk(u);
			float z = sqrt(max(0.f, 1 - d.x * d.x - d.y * d.y));
			return { d.x, d.y, z };
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

	}
}

KRR_NAMESPACE_END