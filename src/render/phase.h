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

	KRR_CALLABLE PhaseFunctionSample sample(const Vector3f &wo, const Vector2f &u) const {
		float pdf;
		Vector3f wi = sampleHenveyGreenstein(wo, g, u, &pdf);
		return PhaseFunctionSample{wi, pdf, pdf};
	}

	KRR_CALLABLE float pdf(const Vector3f& wo, const Vector3f& wi) const { return p(wo, wi); }

	KRR_CALLABLE float p(const Vector3f &wo, const Vector3f &wi) const {
		return evalHenveyGreenstein(wo.dot(wi), g);
	}

	KRR_CALLABLE static float evalHenveyGreenstein(float cosTheta, float g);

	KRR_CALLABLE static Vector3f sampleHenveyGreenstein(const Vector3f &wo, float g,
														const Vector2f &u, float *pdf = nullptr);

private: 
	float g;
};

class PhaseFunction : public TaggedPointer<HGPhaseFunction> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE PhaseFunctionSample sample(const Vector3f &wo, const Vector2f &u) const {
		auto sample = [&](auto ptr) -> PhaseFunctionSample { return ptr->sample(wo, u); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(const Vector3f &wo, const Vector3f &wi) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdf(wo, wi); };
		return dispatch(pdf);
	}

	KRR_CALLABLE float p(const Vector3f &wo, const Vector3f &wi) const {
		auto p = [&](auto ptr) -> float { return ptr->p(wo, wi); };
		return dispatch(p);
	}
};

KRR_NAMESPACE_END