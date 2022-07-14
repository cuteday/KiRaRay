#pragma once

#include "common.h"
#include "math/math.h"

#define _DEFINE_BSDF_INTERNAL_ROUTINES(bsdf_name)														\
	KRR_CALLABLE static BSDFSample sampleInternal(const ShadingData &sd, vec3f wo, Sampler & sg) {	\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.sample(wo, sg);																		\
	}																									\
																										\
	KRR_CALLABLE static vec3f fInternal(const ShadingData& sd, vec3f wo, vec3f wi) {					\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.f(wo, wi);																			\
	}																									\
																										\
	KRR_CALLABLE static float pdfInternal(const ShadingData& sd, vec3f wo, vec3f wi) {				\
		bsdf_name bsdf;																					\
		bsdf.setup(sd);																					\
		return bsdf.pdf(wo, wi);																		\
	}	

KRR_NAMESPACE_BEGIN

namespace bsdf{
	constexpr float minCosTheta = 1e-6f;
	constexpr float epsilon = 1e-6f;
	constexpr float oneMinusEpsilon = 1.f - epsilon;

	KRR_CALLABLE bool SameHemisphere(const vec3f& w, const vec3f& wp) {
		return w[2] * wp[2] > 0;
	}
	KRR_CALLABLE vec3f ToSameHemisphere(const vec3f& w, const vec3f& wp) {
		return { w[0], w[1], w[2] * wp[2] > 0 ? w[2] : -w[2] };
	}
	KRR_CALLABLE vec3f FaceForward(const vec3f& w, const vec3f& wp) {
		return dot(w, wp) > 0 ? w : -w;
	}

	KRR_CALLABLE float CosTheta(const vec3f& w) { return w[2]; }
	KRR_CALLABLE float Cos2Theta(const vec3f& w) { return w[2] * w[2]; }
	KRR_CALLABLE float AbsCosTheta(const vec3f& w) { return fabs(w[2]); }
	KRR_CALLABLE float Sin2Theta(const vec3f& w) {
		return max((float)0, (float)1 - Cos2Theta(w));
	}
	KRR_CALLABLE float AbsDot(const vec3f& a, const vec3f& b) { return fabs(dot(a, b)); }

	KRR_CALLABLE float SinTheta(const vec3f& w) { return sqrt(Sin2Theta(w)); }

	KRR_CALLABLE float TanTheta(const vec3f& w) { return SinTheta(w) / CosTheta(w); }

	KRR_CALLABLE float Tan2Theta(const vec3f& w) {
		return Sin2Theta(w) / Cos2Theta(w);
	}

	KRR_CALLABLE float CosPhi(const vec3f& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 1 : clamp(w[0] / sinTheta, -1.f, 1.f);
	}

	KRR_CALLABLE float SinPhi(const vec3f& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 0 : clamp(w[1] / sinTheta, -1.f, 1.f);
	}

	KRR_CALLABLE float Cos2Phi(const vec3f& w) { return CosPhi(w) * CosPhi(w); }

	KRR_CALLABLE float Sin2Phi(const vec3f& w) { return SinPhi(w) * SinPhi(w); }

	KRR_CALLABLE float CosDPhi(const vec3f& wa, const vec3f& wb) {
		float waxy = wa[0] * wa[0] + wa[1] * wa[1];
		float wbxy = wb[0] * wb[0] + wb[1] * wb[1];
		if (waxy == 0 || wbxy == 0)
			return 1;
		return clamp((wa[0] * wb[0] + wa[1] * wb[1]) / sqrt(waxy * wbxy), -1.f, 1.f);
	}

	KRR_CALLABLE vec3f Reflect(const vec3f& wo, const vec3f& n) {
		return -wo + 2 * dot(wo, n) * n;
	}

	// eta: etaI/etaT when incident ray (flipped )
	KRR_CALLABLE bool Refract(const vec3f& wi, const vec3f& n, float eta,
		vec3f* wt) {
		// Compute $\cos \theta_\roman{t}$ using Snell's law
		float cosThetaI = dot(n, wi);
		float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
		float sin2ThetaT = eta * eta * sin2ThetaI;

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = sqrt(1 - sin2ThetaT);
		*wt = - eta * wi + (eta * cosThetaI - cosThetaT) * n;
		return true;
	}


}



KRR_NAMESPACE_END