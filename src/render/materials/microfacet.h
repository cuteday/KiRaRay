#pragma once

#include "common.h"
#include "math/math.h"
#include "matutils.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;

// GGX
class GGXMicrofacetDistribution { 
public:
    // GGXMicrofacetDistribution Public Methods
    
    GGXMicrofacetDistribution() = default;

    __both__ static inline float RoughnessToAlpha(float roughness);
    
    __both__ void setup(float ax, float ay, bool samplevis = true){
        sampleVisibleArea = samplevis;
        alphax = max(float(0.001), ax);
        alphay = max(float(0.001), ay);
    }

    __both__ GGXMicrofacetDistribution(float alphax, float alphay,
        bool samplevis = true)
        : sampleVisibleArea(samplevis),
        alphax(max(float(0.001), alphax)),
        alphay(max(float(0.001), alphay)) {}

    __both__ float G1(const vec3f& w) const {
        return 1 / (1 + Lambda(w));
    }
    __both__ float G(const vec3f& wo, const vec3f& wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    __both__ inline float D(const vec3f& wh) const {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.;
        const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        float e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) * tan2Theta;
        //printf("cos2Phi: %f, sin2Phi %f, tan2Theta: %f, cos4Theta: %f\n"
        //    "alphax: %f, alphay: %f, e : %f\n", 
        //    Cos2Phi(wh), Sin2Phi(wh), tan2Theta, cos4Theta, alphax, alphay, e);
        return 1 / (M_PI * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
    }

    __both__ inline vec3f Sample(const vec3f& wo, const vec2f& u) const;
    __both__ inline float Pdf(const vec3f& wo, const vec3f& wh) const;

private:
    // GGXMicrofacetDistribution Private Methods
    __both__ float Lambda(const vec3f& w) const;

    // GGXMicrofacetDistribution Private Data
    float alphax, alphay;
    bool sampleVisibleArea;
};

inline float GGXMicrofacetDistribution::RoughnessToAlpha(float roughness) {
    roughness = max(roughness, (float)1e-3);
    float x = log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
        0.000640711f * x * x * x * x;
}

inline float GGXMicrofacetDistribution::Lambda(const vec3f& w) const {
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;
    // Compute _alpha_ for direction _w_
    float alpha =
        sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__both__ inline static void GGXSample11(float cosTheta, float U1, float U2,
    float* slope_x, float* slope_y) {
    // special case (normal incidence)
    if (cosTheta > .9999) {
        float r = sqrt(U1 / (1 - U1));
        float phi = 6.28318530718 * U2;
        *slope_x = r * cos(phi);
        *slope_y = r * sin(phi);
        return;
    }

    float sinTheta =
        sqrt(max((float)0, (float)1 - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1 / tanTheta;
    float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

    // sample slope_x
    float A = 2 * U1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10;
    float B = tanTheta;
    float D = sqrt(
        max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    }
    else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);

    CHECK(!isinf(*slope_y));
    CHECK(!isnan(*slope_y));
}

__both__ inline static vec3f GGXSample(const vec3f& wi, float alpha_x,
    float alpha_y, float U1, float U2) {
    // 1. stretch wi
    vec3f wiStretched =
        normalize(vec3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    GGXSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

    // 3. rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return normalize(vec3f(-slope_x, -slope_y, 1.));
}

inline vec3f GGXMicrofacetDistribution::Sample(const vec3f& wo,
    const vec2f& u) const {
    vec3f wh;
    if (!sampleVisibleArea) {
        float cosTheta = 0, phi = (2 * M_PI) * u[1];
        if (alphax == alphay) {
            float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
            cosTheta = 1 / sqrt(1 + tanTheta2);
        }
        else {
            phi =
                atan(alphay / alphax * tan(2 * M_PI * u[1] + .5f * M_PI));
            if (u[1] > .5f) phi += M_PI;
            float sinPhi = sin(phi), cosPhi = cos(phi);
            const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            const float alpha2 =
                1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
            cosTheta = 1 / sqrt(1 + tanTheta2);
        }
        float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        wh = sphericalToCartesian(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
    }
    else {
        bool flip = wo.z < 0;
        wh = GGXSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if (flip) wh = -wh;
    }
    return wh;
}

float GGXMicrofacetDistribution::Pdf(const vec3f& wo, const vec3f& wh) const {
    if (sampleVisibleArea)
        return D(wh) * G1(wo) * fabs(dot(wo, wh)) / AbsCosTheta(wo);
    else
        return D(wh) * AbsCosTheta(wh);
}

KRR_NAMESPACE_END