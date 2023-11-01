#pragma once

#include "common.h"
#include "bxdf.h"

#define _DEFINE_BSDF_INTERNAL_ROUTINES(bsdf_name)                                                  \
	KRR_CALLABLE static BSDFSample sampleInternal(const SurfaceInteraction &intr, Vector3f wo,     \
												  Sampler &sg,                                     \
												  TransportMode mode = TransportMode::Radiance) {  \
		bsdf_name bsdf;                                                                            \
		bsdf.setup(intr);                                                                          \
		return bsdf.sample(wo, sg, mode);                                                          \
	}                                                                                              \
                                                                                                   \
	KRR_CALLABLE static SampledSpectrum fInternal(const SurfaceInteraction &intr, Vector3f wo,     \
												  Vector3f wi,                                     \
												  TransportMode mode = TransportMode::Radiance) {  \
		bsdf_name bsdf;                                                                            \
		bsdf.setup(intr);                                                                          \
		return bsdf.f(wo, wi, mode);                                                               \
	}                                                                                              \
                                                                                                   \
	KRR_CALLABLE static float pdfInternal(const SurfaceInteraction &intr, Vector3f wo,             \
										  Vector3f wi,                                             \
										  TransportMode mode = TransportMode::Radiance) {          \
		bsdf_name bsdf;                                                                            \
		bsdf.setup(intr);                                                                          \
		return bsdf.pdf(wo, wi, mode);                                                             \
	}                                                                                              \
                                                                                                   \
	KRR_CALLABLE static BSDFType flagsInternal(const SurfaceInteraction &intr) {                   \
		bsdf_name bsdf;                                                                            \
		bsdf.setup(intr);                                                                          \
		return bsdf.flags();                                                                       \
	}

KRR_NAMESPACE_BEGIN



namespace bsdf{
	constexpr float minCosTheta = 1e-6f;
	constexpr float epsilon = 1e-6f;
	constexpr float oneMinusEpsilon = 1.f - epsilon;

	KRR_CALLABLE bool SameHemisphere(const Vector3f& w, const Vector3f& wp) {
		return w[2] * wp[2] > 0;
	}
	KRR_CALLABLE Vector3f ToSameHemisphere(const Vector3f& w, const Vector3f& wp) {
		return { w[0], w[1], w[2] * wp[2] > 0 ? w[2] : -w[2] };
	}
	KRR_CALLABLE Vector3f FaceForward(const Vector3f& w, const Vector3f& wp) {
		return dot(w, wp) > 0 ? w : -w;
	}

	KRR_CALLABLE float CosTheta(const Vector3f& w) { return w[2]; }
	KRR_CALLABLE float Cos2Theta(const Vector3f& w) { return w[2] * w[2]; }
	KRR_CALLABLE float AbsCosTheta(const Vector3f& w) { return fabs(w[2]); }
	KRR_CALLABLE float Sin2Theta(const Vector3f& w) {
		return max((float)0, (float)1 - Cos2Theta(w));
	}
	KRR_CALLABLE float AbsDot(const Vector3f& a, const Vector3f& b) { return fabs(dot(a, b)); }

	KRR_CALLABLE float SinTheta(const Vector3f& w) { return sqrt(Sin2Theta(w)); }

	KRR_CALLABLE float TanTheta(const Vector3f& w) { return SinTheta(w) / CosTheta(w); }

	KRR_CALLABLE float Tan2Theta(const Vector3f& w) {
		return Sin2Theta(w) / Cos2Theta(w);
	}

	KRR_CALLABLE float CosPhi(const Vector3f& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 1 : clamp(w[0] / sinTheta, -1.f, 1.f);
	}

	KRR_CALLABLE float SinPhi(const Vector3f& w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 0 : clamp(w[1] / sinTheta, -1.f, 1.f);
	}

	KRR_CALLABLE float Cos2Phi(const Vector3f& w) { return CosPhi(w) * CosPhi(w); }

	KRR_CALLABLE float Sin2Phi(const Vector3f& w) { return SinPhi(w) * SinPhi(w); }

	KRR_CALLABLE float CosDPhi(const Vector3f& wa, const Vector3f& wb) {
		float waxy = wa[0] * wa[0] + wa[1] * wa[1];
		float wbxy = wb[0] * wb[0] + wb[1] * wb[1];
		if (waxy == 0 || wbxy == 0)
			return 1;
		return clamp((wa[0] * wb[0] + wa[1] * wb[1]) / sqrt(waxy * wbxy), -1.f, 1.f);
	}

	KRR_CALLABLE Vector3f Reflect(const Vector3f& wo, const Vector3f& n) {
		return -wo + 2 * dot(wo, n) * n;
	}

	// eta: etaT/etaI when incident ray (relative IoR along ray to the caller)
	/*	 \  /
	 *	  \/   etaI
	 *	------------
	 *	 media etaT
	 */			
	KRR_CALLABLE bool Refract(const Vector3f& wi, const Vector3f& n, float eta,
		Vector3f* wt) {
		// Compute $\cos \theta_\roman{t}$ using Snell's law
		float cosThetaI = dot(n, wi);
		float sin2ThetaI = max(float(0), float(1 - pow2(cosThetaI)));
		float sin2ThetaT = sin2ThetaI / pow2(eta);

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = sqrt(1 - sin2ThetaT);
		*wt				= -wi / eta + (cosThetaI / eta - cosThetaT) * n;
		return true;
	}

	// eta: absolute eta (eta_inside / eta_outside), *etap is optional.
	KRR_CALLABLE bool Refract(const Vector3f &wi, Vector3f &n, float eta, float* etap, Vector3f *wt) {
		// Compute $\cos \theta_\roman{t}$ using Snell's law
		float cosThetaI	 = dot(n, wi);
		// Potentially flip interface orientation for Snell's law
		if (wi[2] < 0) eta = 1 / eta;
		float sin2ThetaI = max(float(0), float(1 - pow2(cosThetaI)));
		float sin2ThetaT = sin2ThetaI / pow2(eta);

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = sqrt(1 - sin2ThetaT);
		*wt				= - wi / eta + (cosThetaI / eta- cosThetaT) * n;
		if (etap) *etap = eta;
		return true;
	}

}



KRR_NAMESPACE_END