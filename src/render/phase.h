#pragma once
#include "common.h"
#include "device/taggedptr.h"
#include "krrmath/math.h"

NAMESPACE_BEGIN(krr)

struct PhaseFunctionSample {
	Vector3f wi;
	float p;
	float pdf;
};

class HGPhaseFunction {
public:
	KRR_CALLABLE HGPhaseFunction() = default;
	KRR_CALLABLE HGPhaseFunction(float g) : g(g) {}

	KRR_HOST_DEVICE PhaseFunctionSample sample(const Vector3f &wo, const Vector2f &u) const;

	KRR_HOST_DEVICE float pdf(const Vector3f &wo, const Vector3f &wi) const;

	KRR_HOST_DEVICE float p(const Vector3f &wo, const Vector3f &wi) const;

private: 
	float g;
};

NAMESPACE_END(krr)