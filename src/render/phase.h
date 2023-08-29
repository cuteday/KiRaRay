#pragma once
#include "common.h"
#include "taggedptr.h"
#include "krrmath/math.h"
#include "util/math_utils.h"
#include "shared.h"

KRR_NAMESPACE_BEGIN

namespace rt {

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

	static float evalHenveyGreenstein(float cosTheta, float g) {
		g			= clamp(g, -.99f, .99f);
		float denom = 1 + pow2(g) + 2 * g * cosTheta;
		return M_INV_4PI * (1 - pow2(g)) / (denom * safe_sqrt(denom));
	}

	static Vector3f sampleHenveyGreenstein(const Vector3f& wo, float g, const Vector2f& u,
		float* pdf = nullptr) {
		g = clamp(g, -.99f, .99f);

		// Compute $\cos\theta$ for Henyey--Greenstein sample
		float cosTheta;
		if (fabs(g) < 1e-3f)
			cosTheta = 1 - 2 * u[0];
		else
			cosTheta = -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * u[0])));

		// Compute direction _wi_ for Henyey--Greenstein sample
		float sinTheta = safe_sqrt(1 - pow2(cosTheta));
		float phi	   = M_2PI * u[1];
		Frame wFrame(wo);
		Vector3f wi	   = wFrame.toWorld(utils::sphericalToCartesian(sinTheta, cosTheta, phi));

		if (pdf) *pdf = evalHenveyGreenstein(cosTheta, g);
		return wi;
	}

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

} // namespace rt

KRR_NAMESPACE_END