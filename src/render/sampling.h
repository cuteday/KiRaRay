#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "common.h"

#define MIS_POWER_HEURISTIC 1	// 0 for balance heuristic

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace utils;

__both__ inline float evalMIS(float p0, float p1) {
#if MIS_POWER_HEURISTIC
	return p0 * p0 / (p0 * p0 + p1 * p1);
#else 
	return p0 / (p0 + p1);
#endif
}

__both__ inline float evalMIS(float n0, float p0, float n1, float p1) {
#if MIS_POWER_HEURISTIC
	float q0 = (n0 * p0) * (n0 * p0);
	float q1 = (n1 * p1) * (n1 * p1);
	return q0 / (q0 + q1);
#else 
	return (n0 * p0) / (n0 * p0 + n1 * p1);
#endif
}

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

__both__ inline vec3f uniformSampleTriangle(const vec2f& u) {
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

//__both__ inline vec3f sampleSphericalTriangle(vec3f v, vec3f p, vec2f u, float& pdf) {
//	if (pdf)
//		pdf = 0;
//	// Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
//	vec3f a(v[0] - p), b(v[1] - p), c(v[2] - p);
//	CHECK_GT(lengthSquared(a), 0);
//	CHECK_GT(lengthSquared(b), 0);
//	CHECK_GT(lengthSquared(c), 0);
//	a = normalize(a);
//	b = normalize(b);
//	c = normalize(c);
//
//	// Compute normalized cross products of all direction pairs
//	vec3f n_ab = cross(a, b), n_bc = cross(b, c), n_ca = cross(c, a);
//	if (lengthSquared(n_ab) == 0 || lengthSquared(n_bc) == 0 || lengthSquared(n_ca) == 0)
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
//	vec3f cp = cosBp * a + sinBp * normalize(GramSchmidt(c, a));
//
//	// Compute sampled spherical triangle direction and return barycentrics
//	float cosTheta = 1 - u[1] * (1 - Dot(cp, b));
//	float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
//	vec3f w = cosTheta * b + sinTheta * normalize(GramSchmidt(cp, b));
//	// Find barycentric coordinates for sampled direction _w_
//	vec3f e1 = v[1] - v[0], e2 = v[2] - v[0];
//	vec3f s1 = cross(w, e2);
//	float divisor = Dot(s1, e1);
//
//	if (divisor == 0) {
//		// This happens with triangles that cover (nearly) the whole
//		// hemisphere.
//		return { 1.f / 3.f, 1.f / 3.f, 1.f / 3.f };
//	}
//	float invDivisor = 1 / divisor;
//	vec3f s = p - v[0];
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

/////////////////////////////////////////////////////////////////////////
//				ggx microfacet evaluation and sampling				   //
/////////////////////////////////////////////////////////////////////////
__both__ inline float evalG1GGX(float alphaSqr, float cosTheta)
{
	if (cosTheta <= 0) return 0;
	float cosThetaSqr = cosTheta * cosTheta;
	float tanThetaSqr = max(1 - cosThetaSqr, 0.f) / cosThetaSqr;
	return 2 / (1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

__both__ inline float evalNdfGGX(float alpha, float cosTheta)
{
	float a2 = alpha * alpha;
	float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
	return a2 / (d * d * M_PI);
}

__both__ inline float evalPdfGGX_VNDF(float alpha, vec3f wo, vec3f h)
{
	float G1 = evalG1GGX(alpha * alpha, wo.z);
	float D = evalNdfGGX(alpha, h.z);
	return G1 * D * max(0.f, dot(wo, h)) / wo.z;
}

__both__ inline float evalLambdaGGX(float alphaSqr, float cosTheta)
{
	if (cosTheta <= 0) return 0;
	float cosThetaSqr = cosTheta * cosTheta;
	float tanThetaSqr = max(1 - cosThetaSqr, 0.f) / cosThetaSqr;
	return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

__both__ inline float evalMaskingSmithGGXSeparable(float alpha, float cosThetaI, float cosThetaO)
{
	float alphaSqr = alpha * alpha;
	float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
	float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
	return 1 / ((1 + lambdaI) * (1 + lambdaO));
}

__both__ inline vec3f sampleGGX_VNDF(float alpha, vec3f wo, vec2f u)
{
	float alpha_x = alpha, alpha_y = alpha;

	// Transform the view vector to the hemisphere configuration.
	vec3f Vh = normalize(vec3f(alpha_x * wo.x, alpha_y * wo.y, wo.z));

	// Construct orthonormal basis (Vh,T1,T2).
	vec3f T1 = (Vh.z < 0.9999f) ? normalize(cross(vec3f(0, 0, 1), Vh)) : vec3f(1, 0, 0); // TODO: fp32 precision
	vec3f T2 = cross(Vh, T1);

	// Parameterization of the projected area of the hemisphere.
	float r = sqrt(u.x);
	float phi = (2.f * M_PI) * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5f * (1.f + Vh.z);
	t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

	// Reproject onto hemisphere.
	vec3f Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

	// Transform the normal back to the ellipsoid configuration. This is our half vector.
	vec3f h = normalize(vec3f(alpha_x * Nh.x, alpha_y * Nh.y, max(0.f, Nh.z)));
	return h;
}

KRR_NAMESPACE_END