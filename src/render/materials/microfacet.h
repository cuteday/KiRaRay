#pragma once

#include "common.h"

#include "render/shared.h"

#include "bxdf.h"
#include "matutils.h"
#include "fresnel.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;
using namespace shader;

/* Trowbridge-Reitz distribution */
class GGXMicrofacetDistribution { 
public:
	GGXMicrofacetDistribution() = default;

	KRR_CALLABLE bool isSpecular() const { return max(alphax, alphay) <= 1e-3f; }

	static KRR_CALLABLE float RoughnessToAlpha(float roughness) {
		roughness = max(roughness, (float)1e-3f);
		float x = log(roughness);
		return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
			0.000640711f * x * x * x * x;
	};

	KRR_CALLABLE GGXMicrofacetDistribution(float alphax, float alphay):
		alphax(max(float(1e-3f), alphax)), 
		alphay(max(float(1e-3f), alphay)) {}

	KRR_CALLABLE float G1(const Vector3f& w) const {
		return 1 / (1 + Lambda(w));
	}

	KRR_CALLABLE float G(const Vector3f& wo, const Vector3f& wi) const {
		return 1 / (1 + Lambda(wo) + Lambda(wi));
	}

	KRR_CALLABLE float D(const Vector3f& wh) const {
		float tan2Theta = Tan2Theta(wh);
		if (isinf(tan2Theta)) return 0.;
		const float cos4Theta = pow2(Cos2Theta(wh));
		float e = (pow2(CosPhi(wh) / alphax) + pow2(SinPhi(wh) / alphay)) * tan2Theta;
		return 1 / (M_PI * alphax * alphay * cos4Theta * pow2(1 + e));
	}

	KRR_CALLABLE Vector3f Sample(const Vector3f& wo, const Vector2f& u) const;
	
	KRR_CALLABLE float Pdf(const Vector3f& wo, const Vector3f& wh) const {
		return D(wh) * G1(wo) * fabs(dot(wo, wh)) / AbsCosTheta(wo);
	};

private:
	// GGXMicrofacetDistribution Private Methods
	KRR_CALLABLE float Lambda(const Vector3f& w) const {
		float absTanTheta = abs(TanTheta(w));
		if (isinf(absTanTheta)) return 0.;
		// Compute _alpha_ for direction _w_
		float alpha =
			sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
		float alpha2Tan2Theta = pow2(alpha * absTanTheta);
		return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
	};

	// GGXMicrofacetDistribution Private Data
	float alphax, alphay;
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
	if (cosTheta > .9999f) {
		float r = sqrt(U1 / (1 - U1));
		float phi = M_2PI * U2;
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

/* @returns: sampled microfacet normal that always face towards wo. */
inline Vector3f GGXMicrofacetDistribution::Sample(const Vector3f& wo,
	const Vector2f& u) const {
#define KRR_GGX_SAMPLE_LEGACY
	
#ifdef KRR_GGX_SAMPLE_LEGACY
	bool flip = wo[2] < 0;
	Vector3f wh = GGXSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
	if (flip) wh = -wh;
	return wh;
#else
	Vector3f wh = normalize(Vector3f(alphax * wo[0], alphay * wo[1], wo[2]));
	if (wh[2] < 0)
		wh = -wh;

	// Find orthonormal basis for visible normal sampling
	Vector3f T1 = (wh[2] < 0.99999f) ? normalize(cross(Vector3f(0, 0, 1), wh)) 
		: Vector3f(1, 0, 0);
	Vector3f T2 = cross(wh, T1);

	// Generate uniformly distributed points on the unit disk
	Point2f p = uniformSampleDiskPolar(u);

	// Warp hemispherical projection for visible normal sampling
	float h = std::sqrt(1 - pow2(p[0]));
	p[1]		= lerp((1 + wh[2]) / 2, h, p[1]);

	// Reproject to hemisphere and transform normal to ellipsoid configuration
	float pz	= sqrt(max(0.f, 1.f - Vector2f(p).squaredNorm()));
	Vector3f nh = p[0] * T1 + p[1] * T2 + pz * wh;
	return normalize(Vector3f(alphax * nh[0], alphay * nh[1], max(1e-6f, nh[2])));
#endif
}

class MicrofacetBrdf {
public:
	MicrofacetBrdf() = default;

	KRR_CALLABLE MicrofacetBrdf(const Color &R, float eta, float alpha_x, float alpha_y)
		:R(R), eta(eta){
		distribution = { alpha_x, alpha_y };
	}

	KRR_CALLABLE void setup(const ShadingData &sd) {
		R			 = sd.specular;
		float alpha	 = pow2(sd.roughness);
		eta			 = sd.IoR;
		distribution = { alpha, alpha };
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		if (distribution.isSpecular()) return 0;
		if (!SameHemisphere(wi, wo)) return 0;
		
		float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
		Vector3f wh = wi + wo;
		if (cosThetaI == 0 || cosThetaO == 0) return Vector3f::Zero();
		if (!any(wh)) return 0;
		wh = normalize(wh);
		Color F = Fr(wo, wh);

		return distribution.D(wh) * distribution.G(wo, wi) * F /
			(4 * cosThetaI * cosThetaO);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler& sg) const {
		BSDFSample sample = {};
		Vector3f wi, wh;
		Vector2f u = sg.get2D();

		if (wo[2] == 0) return sample;     
		if (distribution.isSpecular()) {
			return BSDFSample(Fr(wo, { 0, 0, 1 }) / AbsCosTheta(wo), 
				{ -wo[0], -wo[1], wo[2] }, 
				1 /* delta pdf */, BSDF_SPECULAR_REFLECTION /* bsdf type */);
		}
		
		wh = distribution.Sample(wo, u);
		if (dot(wo, wh) < 0) return sample;

		wi = Reflect(wo, wh);
		if (!SameHemisphere(wo, wi)) return sample;

		// Compute PDF of _wi_ for microfacet reflection
		sample.f = f(wo, wi);
		sample.wi = wi;
		sample.pdf = distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
		sample.flags = BSDF_GLOSSY_REFLECTION;
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		if (distribution.isSpecular()) return 0;
		if (!SameHemisphere(wo, wi)) return 0;
		Vector3f wh = normalize(wo + wi);
		return distribution.Pdf(wo, wh) / (4 * dot(wo, wh));
	}

	KRR_CALLABLE Color Fr(Vector3f wo, Vector3f wh) const {
		// fresnel is also on the microfacet (wrt to wh)
#if KRR_USE_DISNEY
		return DisneyFresnel(disneyR0, metallic, eta, dot(wo, wh)) *
			   R; // etaT / etaI, auto inversion.
#else
		return FrSchlick(R, Color3f(1.f), dot(wo, wh));
		return FrDielectric(dot(wo, wh), eta); // etaT / etaI.
#endif
	}

	KRR_CALLABLE BSDFType flags() const {
		if (!R.any()) return BSDF_UNSET;
		return distribution.isSpecular() ? BSDF_SPECULAR_REFLECTION : BSDF_GLOSSY_REFLECTION;
	}
	
	Color R{ 1 };	            // specular reflectance
	float eta{ 1.5 };	        // eta_inner / eta_outer

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

	KRR_CALLABLE MicrofacetBtdf(const Color &T, float eta, float alpha_x, float alpha_y)
		: T(T), etaT(eta) {
		distribution = { alpha_x, alpha_y };
	}

	KRR_CALLABLE void setup(const ShadingData& sd) {
		T = sd.diffuse * sd.specularTransmission;
		float alpha = pow2(sd.roughness);
		// TODO: ETA=1 causes NaN, should switch to delta scattering
		etaT = max(1.01f, sd.IoR);
		distribution = { alpha, alpha };
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		if (distribution.isSpecular()) return 0;
		if (SameHemisphere(wo, wi)) return 0;

		float cosThetaO = wo[2], cosThetaI = wi[2];
		if (cosThetaI == 0 || cosThetaO == 0) return 0;

		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		float eta = CosTheta(wo) > 0 ? etaT : 1 / etaT;
		Vector3f wh = normalize(wo + wi * eta);
		if (wh[2] < 0) wh = -wh;

		// Same side?
		if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
		Color F = Fr(wo, wh);
		
		float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);

		return (Color::Ones() - F) * T *
			fabs(distribution.D(wh) * distribution.G(wo, wi) *
				AbsDot(wi, wh) * AbsDot(wo, wh)  /
				(cosThetaI * cosThetaO * pow2(sqrtDenom)));
	}

	KRR_CALLABLE  BSDFSample sample(Vector3f wo, Sampler& sg) const {
		BSDFSample sample = {};
		if (wo[2] == 0) return sample;

		float eta;
		if (distribution.isSpecular()) {
			Vector3f wi, wh = { 0, 0, copysignf(1, wo[2]) };
			if (!Refract(wo, wh, etaT, &eta, &wi))
				return {};
			
			Color ft = (Color::Ones() - Fr(wo, wh)) * T / AbsCosTheta(wi);
			return BSDFSample(ft, wi, 1, BSDF_SPECULAR_TRANSMISSION);
		}

		Vector2f u = sg.get2D();
		Vector3f wh = distribution.Sample(wo, u);
		
		if (!Refract(wo, wh, etaT, &eta, &sample.wi)) return {};   // etaI / etaT

		sample.pdf = pdf(wo, sample.wi);
		sample.f = f(wo, sample.wi);
		sample.flags = BSDF_GLOSSY_TRANSMISSION;
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		if (distribution.isSpecular()) return 0;
		if (SameHemisphere(wo, wi)) return 0;
		float eta = CosTheta(wo) > 0 ? etaT : 1 / etaT;
		// wh = wo + eta * wi, eta = etaI / etaT
		Vector3f wh = normalize(wo + wi * eta);	// eta=etaI/etaT

		if (dot(wo, wh) * dot(wi, wh) > 0) return 0;
		// Compute change of variables _dwh\_dwi_ for microfacet transmission
		float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
		float dwh_dwi = fabs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
		return distribution.Pdf(wo, wh) * dwh_dwi;
	}

	KRR_CALLABLE BSDFType flags() const {
		if (!T.any()) return BSDF_UNSET;
		return distribution.isSpecular() ? BSDF_SPECULAR_TRANSMISSION : BSDF_GLOSSY_TRANSMISSION;
	}

	KRR_CALLABLE Color Fr(Vector3f wo, Vector3f wh) const {
#if KRR_USE_DISNEY
		return DisneyFresnel(disneyR0, metallic, etaT, dot(wo, wh)); // etaT / etaI, auto inversion
#else
		return FrSchlick(R, Vector3f(1.f), dot(wo, wh)) / R;
		return FrDielectric(dot(wo, wh), wo[2]), etaT);
#endif
	}

	Color T{ 0 }; // specular reflectance
	float etaT{ 1.5 }				/* etaA: outside IoR, etaB: inside IoR */;
#if KRR_USE_DISNEY
	Color disneyR0;
	float metallic;
	DisneyMicrofacetDistribution distribution; // separable masking shadow model for disney
#else
	GGXMicrofacetDistribution distribution;
#endif
};

KRR_NAMESPACE_END