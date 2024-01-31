#pragma once

#include "common.h"

#include "util/check.h"
#include "util/hash.h"

NAMESPACE_BEGIN(krr)

namespace utils {
/*******************************************************
 * Numerical
 ********************************************************/
template <typename C>
KRR_CALLABLE constexpr float evaluatePolynomial(float t, C c) {
	return c;
}

template <typename C, typename ...Args>
KRR_CALLABLE constexpr float evaluatePolynomial(float t, C c, Args... cRemaining) {
	return fma(t, evaluatePolynomial(t, cRemaining...), c);
}

template <class To, class From>
KRR_CALLABLE
	typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
								  std::is_trivially_copyable_v<To>, To>
	bit_cast(const From &src) noexcept {
	static_assert(std::is_trivially_constructible_v<To>,
				  "This implementation requires the destination type to be trivially "
				  "constructible");
	To dst;
	std::memcpy(&dst, &src, sizeof(To));
	return dst;
}

KRR_CALLABLE uint64_t floatToBits(double f) {
#ifdef KRR_DEVICE_CODE
	return __double_as_longlong(f);
#else
	return bit_cast<uint64_t>(f);
#endif
}

KRR_CALLABLE double bitsToFloat(uint64_t ui) {
#ifdef KRR_DEVICE_CODE
	return __longlong_as_double(ui);
#else
	return bit_cast<double>(ui);
#endif
}

KRR_CALLABLE float nextFloatUp(float v) {
	// Handle infinity and negative zero for _NextFloatUp()_
	if (isinf(v) && v > 0.f)
		return v;
	if (v == -0.f) v = 0.f;
	// Advance _v_ to next higher float
	uint32_t ui = floatToBits(v);
	if (v >= 0) ++ui;
	else --ui;
	return bitsToFloat(ui);
}

KRR_CALLABLE float nextFloatDown(float v) {
	// Handle infinity and positive zero for _NextFloatDown()_
	if (isinf(v) && v < 0.f)
		return v;
	if (v == 0.f) v = -0.f;
	uint32_t ui = floatToBits(v);
	if (v > 0) --ui;
	else ++ui;
	return bitsToFloat(ui);
}

/*******************************************************
 * algorithms
 ********************************************************/
template <typename T> KRR_CALLABLE void extendedGCD(T a, T b, T *x, T *y) {
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

template <typename T> KRR_CALLABLE T multiplicativeInverse(T a, T n) {
	T x, y;
	extendedGCD(a, n, &x, &y);
	return x % n;
}

template <typename Predicate>
KRR_CALLABLE size_t findInterval(size_t sz, const Predicate &pred) {
	using ssize_t = std::make_signed_t<size_t>;
	ssize_t size = (ssize_t) sz - 2, first = 1;
	while (size > 0) {
		// Evaluate predicate at midpoint and update _first_ and _size_
		size_t half = (size_t) size >> 1, middle = first + half;
		bool predResult = pred(middle);
		first			= predResult ? middle + 1 : first;
		size			= predResult ? size - (half + 1) : half;
	}
	return (size_t) clamp((ssize_t) first - 1, 0, sz - 2);
}

/*******************************************************
 * vectors and coordinates
 ********************************************************/

KRR_CALLABLE float sphericalTriangleArea(Vector3f a, Vector3f b, Vector3f c) {
	return abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)));
}

// generate a perpendicular vector which is orthogonal to the given vector
KRR_CALLABLE Vector3f getPerpendicular(const Vector3f &u) {
	Vector3f a	 = abs(u);
	uint32_t uyx = (a[0] - a[1]) < 0 ? 1 : 0;
	uint32_t uzx = (a[0] - a[2]) < 0 ? 1 : 0;
	uint32_t uzy = (a[1] - a[2]) < 0 ? 1 : 0;
	uint32_t xm	 = uyx & uzx;
	uint32_t ym	 = (1 ^ xm) & uzy;
	uint32_t zm	 = 1 ^ (xm | ym); // 1 ^ (xm & ym)
	Vector3f v	 = normalize(cross(u, Vector3f(xm, ym, zm)));
	return v;
}

// world => y-up
KRR_CALLABLE Vector2f worldToLatLong(const Vector3f &dir) {
	Vector3f p = normalize(dir);
	Vector2f uv;
	uv[0] = atan2(p[0], -p[2]) * M_INV_2PI + 0.5f;
	uv[1] = acos(p[1]) * M_INV_PI;
	return uv;
}

/// <param name="latlong"> in [0, 1]*[0, 1] </param>
KRR_CALLABLE Vector3f latlongToWorld(Vector2f latlong) {
	float phi	   = M_PI * (2.f * saturate(latlong[0]) - 1.f);
	float theta	   = M_PI * saturate(latlong[1]);
	float sinTheta = sin(theta);
	float cosTheta = cos(theta);
	float sinPhi   = sin(phi);
	float cosPhi   = cos(phi);
	return { sinTheta * sinPhi, cosTheta, -sinTheta * cosPhi };
}

// caetesian, or local frame => z-up
// @returns
//	theta: [0, pi], phi: [0, 2pi]
KRR_CALLABLE Vector2f cartesianToSpherical(const Vector3f &v) {
	/* caution! acos(val) produces NaN when val is out of [-1, 1]. */
	const Vector3f vn = v.normalized();
	Vector2f sph{ acos(vn[2]), atan2(vn[1], vn[0]) };
	if (sph[1] < 0)
		sph[1] += M_2PI;
	return sph;
}

KRR_CALLABLE Vector2f cartesianToSphericalNormalized(const Vector3f &v) {
	Vector2f sph = cartesianToSpherical(v);
	return { float(sph[0] * M_INV_PI), float(sph[1] * M_INV_2PI) };
}

/// <param name="sph">\phi in [0, 2pi] and \theta in [0, pi]</param>
KRR_CALLABLE Vector3f sphericalToCartesian(float theta, float phi) {
	float sinTheta = sin(theta);
	float cosTheta = cos(theta);
	float sinPhi   = sin(phi);
	float cosPhi   = cos(phi);
	return {
		sinTheta * cosPhi,
		sinTheta * sinPhi,
		cosTheta,
	};
}

KRR_CALLABLE Vector3f sphericalToCartesian(float sinTheta, float cosTheta, float phi) {
	return Vector3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

KRR_CALLABLE Vector3f sphericalToCartesian(const Vector2f sph) {
	return sphericalToCartesian(sph[0], sph[1]);
}

KRR_CALLABLE float sphericalTheta(const Vector3f &v) { return acos(clamp(v[2], -1.f, 1.f)); }

KRR_CALLABLE float sphericalPhi(const Vector3f &v) {
	float p = atan2(v[1], v[0]);
	return (p < 0) ? (p + 2 * M_PI) : p;
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
	uint64_t n0 = ReverseBits32((uint32_t) n);
	uint64_t n1 = ReverseBits32((uint32_t) (n >> 32));
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

KRR_CALLABLE void DecodeMorton2(uint64_t v, uint32_t *x, uint32_t *y) {
	*x = Compact1By1(v);
	*y = Compact1By1(v >> 1);
}

KRR_CALLABLE uint32_t Compact1By2(uint32_t x) {
	x &= 0x09249249;				  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >> 2)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >> 4)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >> 8)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}
} // namespace utils

NAMESPACE_END(krr)