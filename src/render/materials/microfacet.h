#pragma once

#include "common.h"
#include "math/math.h"
#include "render/shared.h"

#include "bxdf.h"
#include "matutils.h"
#include "fresnel.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;
using namespace shader;

// GGX
class GGXMicrofacetDistribution { 
public:
    GGXMicrofacetDistribution() = default;

    KRR_CALLABLE bool isSpecular() { return max(alphax, alphay) < 1e-3f; }

    __both__ static inline float RoughnessToAlpha(float roughness) {
        roughness = max(roughness, (float)1e-3);
        float x = log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
            0.000640711f * x * x * x * x;
    };
    
    __both__ void setup(float ax, float ay, bool samplevis = true){
        sampleVisibleArea = samplevis;
        alphax = max(1e-4f, ax);
        alphay = max(1e-4f, ay);
    }

    __both__ GGXMicrofacetDistribution(float alphax, float alphay,
        bool samplevis = true)
        : sampleVisibleArea(samplevis),
        alphax(max(float(0.001), alphax)),
        alphay(max(float(0.001), alphay)) {}

    __both__ float G1(const Vector3f& w) const {
        return 1 / (1 + Lambda(w));
    }

    __both__ float G(const Vector3f& wo, const Vector3f& wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }

    __both__ float D(const Vector3f& wh) const {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.;
        const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        float e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) * tan2Theta;
        return 1 / (M_PI * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
    }

    KRR_CALLABLE Vector3f Sample(const Vector3f& wo, const Vector2f& u) const;
    
    __both__ float Pdf(const Vector3f& wo, const Vector3f& wh) const {
        if (sampleVisibleArea)
            return D(wh) * G1(wo) * fabs(dot(wo, wh)) / AbsCosTheta(wo);
        else
            return D(wh) * AbsCosTheta(wh);
    };

private:
    // GGXMicrofacetDistribution Private Methods
    KRR_CALLABLE float Lambda(const Vector3f& w) const {
        float absTanTheta = abs(TanTheta(w));
        if (isinf(absTanTheta)) return 0.;
        // Compute _alpha_ for direction _w_
        float alpha =
            sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
        float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
        return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
    };

    // GGXMicrofacetDistribution Private Data
    float alphax, alphay;
    bool sampleVisibleArea;
};


class DisneyMicrofacetDistribution : public GGXMicrofacetDistribution {
public:
    DisneyMicrofacetDistribution() = default;

    KRR_CALLABLE DisneyMicrofacetDistribution(float alphax, float alphay)
        : GGXMicrofacetDistribution(alphax, alphay) {}

    KRR_CALLABLE float G(const Vector3f& wo, const Vector3f& wi) const {
        // Disney uses the separable masking-shadowing model.
        return G1(wo) * G1(wi);
    }
};

KRR_CALLABLE static void GGXSample11(float cosTheta, float U1, float U2,
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

KRR_CALLABLE static Vector3f GGXSample(const Vector3f& wi, float alpha_x,
    float alpha_y, float U1, float U2) {
    // 1. stretch wi
    Vector3f wiStretched =
        normalize(Vector3f(alpha_x * wi[0], alpha_y * wi[1], wi[2]));

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
    return normalize(Vector3f(-slope_x, -slope_y, 1.));
}

inline Vector3f GGXMicrofacetDistribution::Sample(const Vector3f& wo,
    const Vector2f& u) const {
    Vector3f wh;
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
        bool flip = wo[2] < 0;
        wh = GGXSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if (flip) wh = -wh;
    }
    return wh;
}

class MicrofacetBrdf {
public:
    MicrofacetBrdf() = default;

    __both__ MicrofacetBrdf(const Color &R, float eta, float alpha_x, float alpha_y)
        :R(R), eta(eta){
        distribution.setup(alpha_x, alpha_y);
    }

    _DEFINE_BSDF_INTERNAL_ROUTINES(MicrofacetBrdf);

    __both__ void setup(const ShadingData& sd) {
        R = sd.specular;
        float alpha = GGXMicrofacetDistribution::RoughnessToAlpha(sd.roughness);
        alpha = pow2(sd.roughness);
        eta = sd.IoR;
        distribution.setup(alpha, alpha, true);
    }

    __both__ Color f(Vector3f wo, Vector3f wi) const {
        float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
        Vector3f wh = wi + wo;
        if (cosThetaI == 0 || cosThetaO == 0) return Vector3f::Zero();
		if (!any(wh))
			return Vector3f::Zero();
        wh = normalize(wh);

        // fresnel is also on the microfacet (wrt to wh)
#if KRR_USE_DISNEY
		Color F = DisneyFresnel(disneyR0, metallic, eta, dot(wo, wh)); // etaT / etaI
#elif KRR_USE_SCHLICK_FRESNEL
        Vector3f F = FrSchlick(R, Vector3f(1.f), dot(wo, wh)) / R;
#else
        Vector3f F = FrDielectric(dot(wo, wh), eta);	// etaT / etaI
#endif
        return R * distribution.D(wh) * distribution.G(wo, wi) * F /
            (4 * cosThetaI * cosThetaO);
    }

    __both__  BSDFSample sample(Vector3f wo, Sampler& sg) const {
        BSDFSample sample = {};
        Vector3f wi, wh;
        Vector2f u = sg.get2D();

        if (wo[2] == 0) return sample;
        wh = distribution.Sample(wo, u);
        if (dot(wo, wh) < 0) return sample;

        wi = Reflect(wo, wh);
        if (!SameHemisphere(wo, wi)) return sample;

        // Compute PDF of _wi_ for microfacet reflection
        sample.f = f(wo, wi);
        sample.wi = wi;
        sample.pdf = distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
        return sample;

    }

    __both__ float pdf(Vector3f wo, Vector3f wi) const {
        if (!SameHemisphere(wo, wi)) return 0;
        Vector3f wh = normalize(wo + wi);
        return distribution.Pdf(wo, wi) / (4 * dot(wo, wh));
    }

    Color R{ 1 };	  // specular reflectance
    float eta{ 1.5 };									// 

#if KRR_USE_DISNEY
	Color disneyR0;
    float metallic;
    DisneyMicrofacetDistribution distribution;          // separable masking shadow model for disney
#else
    GGXMicrofacetDistribution distribution;
#endif
};

// Microfacet transmission, for dielectric only now.
class MicrofacetBtdf {
public:
    MicrofacetBtdf() = default;

    __both__ MicrofacetBtdf(const Color &T, float etaA, float etaB, float alpha_x, float alpha_y)
        :T(T), etaA(etaA), etaB(etaB) {
        distribution.setup(alpha_x, alpha_y, true);
    }

    _DEFINE_BSDF_INTERNAL_ROUTINES(MicrofacetBtdf);

    __both__ void setup(const ShadingData& sd) {
        T = sd.transmission;
        float alpha = GGXMicrofacetDistribution::RoughnessToAlpha(sd.roughness);
        alpha = pow2(sd.roughness);
        // TODO: ETA=1 causes NaN, should switch to delta scattering
        etaA = 1; etaB = max(1.01f, sd.IoR);
        etaB = 1.1;
        distribution.setup(alpha, alpha, true);
    }

    __both__ Color f(Vector3f wo, Vector3f wi) const {
		if (SameHemisphere(wo, wi))
			return 0;

        float cosThetaO = wo[2], cosThetaI = wi[2];
		if (cosThetaI == 0 || cosThetaO == 0)
			return 0;

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        float eta = CosTheta(wo) > 0 ? etaB / etaA : etaA / etaB;
        Vector3f wh = normalize(wo + wi * eta);
        if (wh[2] < 0) wh = -wh;

        // Same side?
        if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
		Color F = FrDielectric(dot(wo, wh), etaB / etaA);

        float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        float factor = 1.f / eta;

        return (Color::Ones() - F) * T *
            fabs(distribution.D(wh) * distribution.G(wo, wi) * eta * eta *
                AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
                (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
    }

    __both__  BSDFSample sample(Vector3f wo, Sampler& sg) const {
        BSDFSample sample = {};
        if (wo[2] == 0) return sample;

        Vector2f u = sg.get2D();
        Vector3f wh = distribution.Sample(wo, u);
        if (dot(wo, wh) < 0)
            return sample;  // Should be rare

        float eta = CosTheta(wo) > 0 ? etaA / etaB : etaB / etaA;
        if (!Refract(wo, wh, eta, &sample.wi)) return sample;   // etaI / etaT

        sample.pdf = pdf(wo, sample.wi);
        sample.f = f(wo, sample.wi);
        return sample;
    }

    __both__ float pdf(Vector3f wo, Vector3f wi) const {
        if (SameHemisphere(wo, wi)) return 0;
        float eta = CosTheta(wo) > 0 ? etaB / etaA : etaA / etaB;	// wo is outside?
        // wh = wo + eta * wi, eta = etaI / etaT
        Vector3f wh = normalize(wo + wi * eta);	// eta=etaI/etaT

        if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        float dwh_dwi = fabs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
        float pdf = distribution.Pdf(wo, wh);
        //printf("wh pdf: %.6f, dwh_dwi: %.6f\n", pdf, dwh_dwi);
        return pdf * dwh_dwi;
    }

    Color T{ 0 }; // specular reflectance
    float etaA{ 1 }, etaB{ 1.5 }				/* etaA: outside IoR, etaB: inside IoR */;
    GGXMicrofacetDistribution distribution;
};

KRR_NAMESPACE_END