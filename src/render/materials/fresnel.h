#pragma once

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

namespace bsdf {

    __both__ inline float FrDielectric(float cosTheta_i, float eta) {
        cosTheta_i = Clamp(cosTheta_i, -1, 1);
        // Potentially flip interface orientation for Fresnel equations
        if (cosTheta_i < 0) {
            eta = 1 / eta;
            cosTheta_i = -cosTheta_i;
        }

        // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
        float sin2Theta_i = 1 - Sqr(cosTheta_i);
        float sin2Theta_t = sin2Theta_i / Sqr(eta);
        if (sin2Theta_t >= 1)
            return 1.f;
        float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

        float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
        float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
        return (Sqr(r_parl) + Sqr(r_perp)) / 2;
    }

    __both__ inline float FrComplex(float cosTheta_i, complex<float> eta) {
        using Complex = complex<float>;
        cosTheta_i = Clamp(cosTheta_i, 0, 1);
        // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
        float sin2Theta_i = 1 - Sqr(cosTheta_i);
        Complex sin2Theta_t = sin2Theta_i / Sqr(eta);
        Complex cosTheta_t = sqrt(1 - sin2Theta_t);

        Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
        Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
        return (r_parl.norm() + r_perp.norm()) / 2;
    }

    __both__ inline vec3f FrComplex(float cosTheta_i, vec3f eta,
        vec3f k) {
        vec3f result;
        for (int i = 0; i < NSpectrumSamples; ++i)
            result[i] = FrComplex(cosTheta_i, complex<float>(eta[i], k[i]));
        return result;
    }

}
KRR_NAMESPACE_END