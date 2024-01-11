#pragma once
#include "common.h"

#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

class KumaraswamyDistribution {
public:
	KumaraswamyDistribution() = default;

	KRR_CALLABLE KumaraswamyDistribution(float a, float b) : a(a), b(b) {}

	KRR_CALLABLE float eval(float x) const {
		return a * b * powf(x, a - 1.0f) * powf(1.0f - powf(x, a), b - 1.0f);
	}

	KRR_CALLABLE float pdf(float x) const { return eval(x); }

	KRR_CALLABLE float sample(float u) const {
		return powf(1.0f - powf(1.0f - u, 1.0f / b), 1.0f / a);
	}
	
	float a = 1, b = 1;
};

KRR_NAMESPACE_END