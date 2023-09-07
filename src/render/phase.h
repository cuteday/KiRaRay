#pragma once
#include "common.h"
#include "taggedptr.h"
#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

struct PhaseFunctionSample {
	Vector3f wi;
	float p;
	float pdf;
};

class HGPhaseFunction {
public:
	KRR_CALLABLE HGPhaseFunction() = default;
	KRR_CALLABLE HGPhaseFunction(float g) : g(g) {}

	KRR_CALLABLE PhaseFunctionSample sample(const Vector3f &wo, const Vector2f &u) const;

	KRR_CALLABLE float pdf(const Vector3f &wo, const Vector3f &wi) const;

	KRR_CALLABLE float p(const Vector3f &wo, const Vector3f &wi) const;

private: 
	float g;
};

KRR_NAMESPACE_END