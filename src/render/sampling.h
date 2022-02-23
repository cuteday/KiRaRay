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

/////////////////////////////////////////////////////////////////////////
//				microfacet bsdf evaluation and sampling 

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