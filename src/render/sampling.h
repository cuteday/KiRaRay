#pragma once

#include "common.h"

#include "util/math_utils.h"
#include "interop.h"

#define MIS_POWER_HEURISTIC 1 // 0 for balance heuristic

KRR_NAMESPACE_BEGIN

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

KRR_CALLABLE Vector3f uniformSampleSphere(const Vector2f &u) {
	float z	  = 1.0f - 2.0f * u[0];
	float r	  = sqrt(max(0.0f, 1.0f - z * z));
	float phi = M_2PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

KRR_CALLABLE Vector3f uniformSampleHemisphere(const Vector2f &u) {
	float z	  = u[0];
	float r	  = sqrt(max(0.f, (float) 1.f - z * z));
	float phi = M_2PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

KRR_CALLABLE Vector2f uniformSampleDisk(const Vector2f &u) {
	// the concentric method
	Vector2f uOffset = 2.f * u - Vector2f(1, 1);
	if (uOffset[0] == 0 && uOffset[1] == 0)
		return Vector2f(0, 0);
	float theta, r;
	if (fabs(uOffset[0]) > fabs(uOffset[1])) {
		r	  = uOffset[0];
		theta = M_PI / 4 * (uOffset[1] / uOffset[0]);
	} else {
		r	  = uOffset[1];
		theta = M_PI / 2 - M_PI / 4 * (uOffset[0] / uOffset[1]);
	}
	return r * Vector2f(cos(theta), sin(theta));
}

KRR_CALLABLE Vector2f uniformSampleDiskPolar(const Vector2f &u) {
	float r		= sqrt(u[0]);
	float theta = M_2PI * u[1];
	return Vector2f(r * cos(theta), r * sin(theta));
}


KRR_CALLABLE Vector3f cosineSampleHemisphere(const Vector2f &u) {
	Vector2f d = uniformSampleDisk(u);
	float z	   = sqrt(max(0.f, 1 - d[0] * d[0] - d[1] * d[1]));
	return { d[0], d[1], z };
}

KRR_CALLABLE Vector3f uniformSampleTriangle(const Vector2f &u) {
	float b0, b1;
	if (u[0] < u[1]) {
		b0 = u[0] / 2;
		b1 = u[1] - b0;
	} else {
		b1 = u[1] / 2;
		b0 = u[0] - b1;
	}
	return { b0, b1, 1 - b0 - b1 };
}

KRR_CALLABLE float sampleExponential(float u, float a) { 
	DCHECK_GT(a, 0); 
	return -std::log(1 - u) / a;
}

KRR_CALLABLE int sampleDiscrete(float* weights, size_t n, float u, float *pmf = nullptr) {
	float sumWeights = 0;
	for (int i = 0; i < n; i++) 
		sumWeights += weights[i];

	float up = u * sumWeights;
	if (up == sumWeights) up = nextFloatDown(up);

	int offset = 0;
	float sum  = 0;
	while (sum + weights[offset] <= up) {
		sum += weights[offset++];
		DCHECK_LT(offset, n);
	}
	if (pmf) *pmf = weights[offset] / sumWeights;
	return offset;
}

KRR_CALLABLE int sampleDiscrete(inter::span<const float> weights, float u, float *pmf = nullptr) {
	float sumWeights = 0;
	for (float w : weights) sumWeights += w;

	float up = u * sumWeights;
	if (up == sumWeights) up = nextFloatDown(up);

	int offset = 0;
	float sum  = 0;
	while (sum + weights[offset] <= up) {
		sum += weights[offset++];
		DCHECK_LT(offset, weights.size());
	}
	if (pmf) *pmf = weights[offset] / sumWeights;
	return offset;
}


KRR_NAMESPACE_END