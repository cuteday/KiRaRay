#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "common.h"

#define MIS_POWER_HEURISTIC 1	// 0 for balance heuristic

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace utils;

KRR_CALLABLE float evalMIS(float p0, float p1) {
#if MIS_POWER_HEURISTIC
	return p0 * p0 / (p0 * p0 + p1 * p1);
#else 
	return p0 / (p0 + p1);
#endif
}

KRR_CALLABLE float evalMIS(float n0, float p0, float n1, float p1) {
#if MIS_POWER_HEURISTIC
	float q0 = (n0 * p0) * (n0 * p0);
	float q1 = (n1 * p1) * (n1 * p1);
	return q0 / (q0 + q1);
#else 
	return (n0 * p0) / (n0 * p0 + n1 * p1);
#endif
}

KRR_CALLABLE Vector3f uniformSampleSphere(const Vector2f& u) {
	float z = 1.0f - 2.0f * u[0];
	float r = sqrt(max(0.0f, 1.0f - z * z));
	float phi = 2.0f * M_PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

KRR_CALLABLE Vector3f uniformSampleHemisphere(const Vector2f& u) {
	float z = u[0];
	float r = sqrt(max(0.f, (float)1.f - z * z));
	float phi = 2 * M_PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

KRR_CALLABLE Vector2f uniformSampleDisk(const Vector2f& u) {
	// simpler method derived using marginal distribution...
	//float r = sqrt(u[0]);
	//float theta = 2 * M_PI * u[1];
	//return Vector2f(r * cos(theta), r * sin(theta));
		
	// the concentric method
	Vector2f uOffset = 2.f * u - Vector2f(1, 1);
	if (uOffset[0] == 0 && uOffset[1] == 0)
		return Vector2f(0, 0);
	float theta, r;
	if (fabs(uOffset[0]) > fabs(uOffset[1])) {
		r = uOffset[0];
		theta = M_PI / 4 * (uOffset[1] / uOffset[0]);
	}
	else {
		r = uOffset[1];
		theta = M_PI / 2 - M_PI / 4 * (uOffset[0] / uOffset[1]);
	}
	return r * Vector2f(cos(theta), sin(theta));
}

KRR_CALLABLE Vector3f cosineSampleHemisphere(const Vector2f& u) {
	Vector2f d = uniformSampleDisk(u);
	float z = sqrt(max(0.f, 1 - d[0] * d[0] - d[1] * d[1]));
	return { d[0], d[1], z };
}

KRR_CALLABLE Vector3f uniformSampleTriangle(const Vector2f& u) {
	float b0, b1;
	if (u[0] < u[1]) {
		b0 = u[0] / 2;
		b1 = u[1] - b0;
	}
	else {
		b1 = u[1] / 2;
		b0 = u[0] - b1;
	}
	return { b0, b1, 1 - b0 - b1 };
}

//KRR_CALLABLE Vector3f sampleSphericalTriangle(Vector3f v, Vector3f p, Vector2f u, float& pdf) {
//	if (pdf)
//		pdf = 0;
//	// Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
//	Vector3f a(v[0] - p), b(v[1] - p), c(v[2] - p);
//	CHECK_GT(squaredLength(a), 0);
//	CHECK_GT(squaredLength(b), 0);
//	CHECK_GT(squaredLength(c), 0);
//	a = normalize(a);
//	b = normalize(b);
//	c = normalize(c);
//
//	// Compute normalized cross products of all direction pairs
//	Vector3f n_ab = cross(a, b), n_bc = cross(b, c), n_ca = cross(c, a);
//	if (squaredLength(n_ab) == 0 || squaredLength(n_bc) == 0 || squaredLength(n_ca) == 0)
//		return {};
//	n_ab = normalize(n_ab);
//	n_bc = normalize(n_bc);
//	n_ca = normalize(n_ca);
//
//	// Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
//	float alpha = AngleBetween(n_ab, -n_ca);
//	float beta = AngleBetween(n_bc, -n_ab);
//	float gamma = AngleBetween(n_ca, -n_bc);
//
//	// Uniformly sample triangle area $A$ to compute $A'$
//	float A_pi = alpha + beta + gamma;
//	float Ap_pi = lerp(u[0], M_PI, A_pi);
//	if (pdf) {
//		float A = A_pi - Pi;
//		*pdf = (A <= 0) ? 0 : 1 / A;
//	}
//
//	// Find $\cos \beta'$ for point along _b_ for sampled area
//	float cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);
//	float sinPhi = std::sin(Ap_pi) * cosAlpha - std::cos(Ap_pi) * sinAlpha;
//	float cosPhi = std::cos(Ap_pi) * cosAlpha + std::sin(Ap_pi) * sinAlpha;
//	float k1 = cosPhi + cosAlpha;
//	float k2 = sinPhi - sinAlpha * Dot(a, b) /* cos c */;
//	float cosBp = (k2 + (DifferenceOfProducts(k2, cosPhi, k1, sinPhi)) * cosAlpha) /
//		((SumOfProducts(k2, sinPhi, k1, cosPhi)) * sinAlpha);
//	// Happens if the triangle basically covers the entire hemisphere.
//	// We currently depend on calling code to detect this case, which
//	// is sort of ugly/unfortunate.
//	CHECK(!IsNaN(cosBp));
//	cosBp = Clamp(cosBp, -1, 1);
//
//	// Sample $c'$ along the arc between $b'$ and $a$
//	float sinBp = SafeSqrt(1 - Sqr(cosBp));
//	Vector3f cp = cosBp * a + sinBp * normalize(GramSchmidt(c, a));
//
//	// Compute sampled spherical triangle direction and return barycentrics
//	float cosTheta = 1 - u[1] * (1 - Dot(cp, b));
//	float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
//	Vector3f w = cosTheta * b + sinTheta * normalize(GramSchmidt(cp, b));
//	// Find barycentric coordinates for sampled direction _w_
//	Vector3f e1 = v[1] - v[0], e2 = v[2] - v[0];
//	Vector3f s1 = cross(w, e2);
//	float divisor = Dot(s1, e1);
//
//	if (divisor == 0) {
//		// This happens with triangles that cover (nearly) the whole
//		// hemisphere.
//		return { 1.f / 3.f, 1.f / 3.f, 1.f / 3.f };
//	}
//	float invDivisor = 1 / divisor;
//	Vector3f s = p - v[0];
//	float b1 = Dot(s, s1) * invDivisor;
//	float b2 = Dot(w, cross(s, e1)) * invDivisor;
//
//	// Return clamped barycentrics for sampled direction
//	b1 = Clamp(b1, 0, 1);
//	b2 = Clamp(b2, 0, 1);
//	if (b1 + b2 > 1) {
//		b1 /= b1 + b2;
//		b2 /= b1 + b2;
//	}
//	return { float(1 - b1 - b2), float(b1), float(b2) };
//}

KRR_NAMESPACE_END