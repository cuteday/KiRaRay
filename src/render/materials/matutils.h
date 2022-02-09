#pragma once

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

namespace bsdf{
	constexpr float minCosTheta = 1e-6f;
	constexpr float epsilon = 1e-6f;

    __both__ inline bool SameHemisphere(const vec3f& w, const vec3f& wp) {
        return w.z * wp.z > 0;
    }
    __both__ inline float CosTheta(const vec3f& w) { return w.z; }
    __both__ inline float Cos2Theta(const vec3f& w) { return w.z * w.z; }
    __both__ inline float AbsCosTheta(const vec3f& w) { return abs(w.z); }
    __both__ inline float Sin2Theta(const vec3f& w) {
        return max((float)0, (float)1 - Cos2Theta(w));
    }

    __both__ inline float SinTheta(const vec3f& w) { return sqrt(Sin2Theta(w)); }

    __both__ inline float TanTheta(const vec3f& w) { return SinTheta(w) / CosTheta(w); }

    __both__ inline float Tan2Theta(const vec3f& w) {
        return Sin2Theta(w) / Cos2Theta(w);
    }

    __both__ inline float CosPhi(const vec3f& w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f);
    }

    __both__ inline float SinPhi(const vec3f& w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f);
    }

    __both__ inline float Cos2Phi(const vec3f& w) { return CosPhi(w) * CosPhi(w); }

    __both__ inline float Sin2Phi(const vec3f& w) { return SinPhi(w) * SinPhi(w); }

    __both__ inline float CosDPhi(const vec3f& wa, const vec3f& wb) {
        float waxy = wa.x * wa.x + wa.y * wa.y;
        float wbxy = wb.x * wb.x + wb.y * wb.y;
        if (waxy == 0 || wbxy == 0)
            return 1;
        return clamp((wa.x * wb.x + wa.y * wb.y) / sqrt(waxy * wbxy), -1.f, 1.f);
    }

    __both__ inline vec3f Reflect(const vec3f& wo, const vec3f& n) {
        return -wo + 2 * dot(wo, n) * n;
    }

    __both__ inline bool Refract(const vec3f& wi, const vec3f& n, float eta,
        vec3f* wt) {
        // Compute $\cos \theta_\roman{t}$ using Snell's law
        float cosThetaI = dot(n, wi);
        float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
        float sin2ThetaT = eta * eta * sin2ThetaI;

        // Handle total internal reflection for transmission
        if (sin2ThetaT >= 1) return false;
        float cosThetaT = sqrt(1 - sin2ThetaT);
        *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * vec3f(n);
        return true;
    }

	__both__ inline vec3f evalFresnelSchlick(vec3f f0, vec3f f90, float cosTheta)
	{
		//return lerp(f0, f90, pow(max(1 - cosTheta, 0.f), 5.f));
		return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0.f), 5.f); // clamp to avoid NaN if cosTheta = 1+epsilon
	}

	__both__ inline float evalFresnelSchlick(float f0, float f90, float cosTheta)
	{
		return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0.f), 5.f); // clamp to avoid NaN if cosTheta = 1+epsilon
	}
}
struct BSDFSample {
	vec3f f;
	vec3f wi;
	float pdf = 0;
	uint flags;
	bool valid = true;
};


KRR_NAMESPACE_END