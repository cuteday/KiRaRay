// code taken and modified from pbrt-v3
#pragma once

#include "common.h"

#include "render/shared.h"
#include "math/math.h"
#include "math/utils.h"

#include "bxdf.h"
#include "matutils.h"
#include "fresnel.h"
#include "microfacet.h"

// TODO:
// thin surface not supported
// disney sheen not supported
// disney clearcoat not supported
// subsurface scattering not supported

KRR_NAMESPACE_BEGIN

using namespace shader;
using namespace bsdf;

KRR_CALLABLE float SchlickR0FromEta(float eta) { return pow2(eta - 1) / pow2(eta + 1); }

KRR_CALLABLE float SchlickWeight(float cosTheta) {
	float m = clamp(1.f - cosTheta, 0.f, 1.f);
	return pow5(m);
}

class DisneyDiffuse{
public:
	DisneyDiffuse() = default;

	KRR_CALLABLE DisneyDiffuse(const vec3f& R): R(R) {}
	KRR_CALLABLE vec3f f(const vec3f& wo, const vec3f& wi) const {
		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));

		// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
		// Burley 2015, eq (4).
		return R * M_1_PI * (1 - Fo / 2) * (1 - Fi / 2);
	}
	KRR_CALLABLE vec3f rho(const vec3f&, int, const vec2f*) const { return R; }
	KRR_CALLABLE vec3f rho(int, const vec3f*, const vec3f*) const { return R; }

//private:
	vec3f R;
};

class DisneyFakeSS{
public:
	DisneyFakeSS() = default;

	KRR_CALLABLE DisneyFakeSS(const vec3f& R, float roughness)
		: R(R), roughness(roughness) {}

	KRR_CALLABLE vec3f f(const vec3f& wo, const vec3f& wi) const {
		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return vec3f(0.);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		// Fss90 used to "flatten" retroreflection based on roughness
		float Fss90 = cosThetaD * cosThetaD * roughness;
		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));
		float Fss = lerp(1.f, Fss90, Fo) * lerp(1.f, Fss90, Fi);
		// 1.25 scale is used to (roughly) preserve albedo
		float ss =
			1.25f * (Fss * (1 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - .5f) + .5f);

		return R * M_1_PI * ss;
	};

	KRR_CALLABLE vec3f rho(const vec3f&, int, const vec2f*) const { return R; }
	KRR_CALLABLE vec3f rho(int, const vec2f*, const vec2f*) const { return R; }

//private:
	vec3f R;
	float roughness;
};

class DisneyRetro{
public:
	DisneyRetro() = default;
	KRR_CALLABLE DisneyRetro(const vec3f& R, float roughness)
		: R(R), roughness(roughness) {}
	KRR_CALLABLE vec3f f(const vec3f& wo, const vec3f& wi) const {
		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return vec3f(0.);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));
		float Rr = 2 * roughness * cosThetaD * cosThetaD;

		// Burley 2015, eq (4).
		return R * M_1_PI * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
	};
	KRR_CALLABLE vec3f rho(const vec3f&, int, const vec2f*) const { return R; }
	KRR_CALLABLE vec3f rho(int, const vec2f*, const vec2f*) const { return R; }

//private:
	vec3f R;
	float roughness;
	BxDFType type{ BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE) };
};

class DisneySheen{
public:
	DisneySheen() = default;
	KRR_CALLABLE DisneySheen(const vec3f& R): R(R) {}
	KRR_CALLABLE vec3f f(const vec3f& wo, const vec3f& wi) const {
		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return vec3f(0.);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		return R * SchlickWeight(cosThetaD);
	}

	KRR_CALLABLE vec3f rho(const vec3f&, int, const vec2f*) const { return R; }
	KRR_CALLABLE vec3f rho(int, const vec2f*, const vec2f*) const { return R; }

//private:
	vec3f R;
	BxDFType type{ BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE) };
};

KRR_CALLABLE float GTR1(float cosTheta, float alpha) {
	float alpha2 = alpha * alpha;
	return (alpha2 - 1) /
		(M_PI * log(alpha2) * (1 + (alpha2 - 1) * cosTheta * cosTheta));
}

KRR_CALLABLE float smithG_GGX(float cosTheta, float alpha) {
	float alpha2 = alpha * alpha;
	float cosTheta2 = cosTheta * cosTheta;
	return 1 / (cosTheta + sqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
}

class DisneyClearcoat{
public:
	DisneyClearcoat() = default;

	KRR_CALLABLE DisneyClearcoat(float weight, float gloss)
		: weight(weight), gloss(gloss) {}

	KRR_CALLABLE vec3f f(const vec3f& wo, const vec3f& wi) const {
		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return vec3f(0.);
		wh = normalize(wh);

		float Dr = GTR1(AbsCosTheta(wh), gloss);
		float Fr = FrSchlick(.04f, 1.f, dot(wo, wh));
		// The geometric term always based on alpha = 0.25.
		float Gr =
			smithG_GGX(AbsCosTheta(wo), .25) * smithG_GGX(AbsCosTheta(wi), .25);

		return weight * Gr * Fr * Dr / 4;
	};

	KRR_CALLABLE vec3f Sample_f(const vec3f& wo, vec3f* wi, const vec2f& u,
		float* pdf, BxDFType* sampledType) const {

		if (wo.z == 0) return 0.;

		float alpha2 = gloss * gloss;
		float cosTheta = sqrt(
			max(float(0), (1 - pow(alpha2, 1 - u[0])) / (1 - alpha2)));
		float sinTheta = sqrt(max((float)0, 1 - cosTheta * cosTheta));
		float phi = 2 * M_PI * u[1];
		vec3f wh = sphericalToCartesian(sinTheta, cosTheta, phi);
		if (!SameHemisphere(wo, wh)) wh = -wh;

		*wi = Reflect(wo, wh);
		if (!SameHemisphere(wo, *wi)) return vec3f(0.f);

		*pdf = Pdf(wo, *wi);
		return f(wo, *wi);
	};

	KRR_CALLABLE float Pdf(const vec3f& wo, const vec3f& wi) const {
		if (!SameHemisphere(wo, wi)) return 0;

		vec3f wh = wi + wo;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return 0;
		wh = normalize(wh);

		float Dr = GTR1(AbsCosTheta(wh), gloss);
		return Dr * AbsCosTheta(wh) / (4 * dot(wo, wh));
	};

//private:
	float weight, gloss;
	BxDFType type{ BxDFType(BSDF_REFLECTION | BSDF_GLOSSY) };
};

enum DisneyComponent{
	DISNEY_DIFFUSE					= 1 << 1,
	DISNEY_RETRO					= 1 << 2,
	DISNEY_SPEC_REFLECTION			= 1 << 3,
	DISNEY_SPEC_TRANSMISSION		= 1 << 4,
	//DISNEY_CLEARCOAT,
	//DISNEY_SHEEN,
	//DISNEY_FAKE_SUBSURFACE,
	//DISNEY_SUBSURFACE
};

class DisneyBsdf {
public:
	DisneyBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(DisneyBsdf);

	KRR_CALLABLE void setup(const ShadingData& sd) {
		constexpr bool thin = 0;

		vec3f c = sd.diffuse;
		float metallicWeight = sd.metallic;
		float e = 1.5; //sd.IoR; // Some scene has corrupt IoR so we use all 1.5 instead...
		float strans = sd.specularTransmission;
		float diffuseWeight = (1 - metallicWeight) * (1 - strans);
//		float dt = sd.diffuseTransmission;
		float rough = sd.roughness;
		float lum = luminance(c);
		// normalize lum. to isolate hue+sat
		vec3f Ctint = lum > 0 ? (c / lum) : 1;

		float sheenWeight = 0;
		vec3f Csheen;
		if (sheenWeight > 0) {
			assert(false);
			float stint = 0;
			Csheen = lerp(vec3f(1), Ctint, stint);
		}

		if (diffuseWeight > 0) {
			if (thin) {
				assert(false);
				// Blend between DisneyDiffuse and fake subsurface based on flatness, weight using diffTrans.
				//float flat = 0;
				//disneyDiffuse = DisneyDiffuse(diffuseWeight * (1 - flat) * (1 - dt) * c));
				//disneyFakeSS = DisneyFakeSS(diffuseWeight * flat * (1 - dt) * c, rough);
			}
			else {
				vec3f scatterDistance = 0;
				if (!any(scatterDistance)) {
					// No subsurface scattering; use regular (Fresnel modified) diffuse.
					disneyDiffuse = DisneyDiffuse(diffuseWeight * c);
					components |= DISNEY_DIFFUSE;
				}
				else {
					// TODO: use a BSSRDF instead.
					assert(false);
				}
			}

			disneyRetro = DisneyRetro(diffuseWeight * c, rough);
			components |= DISNEY_RETRO;

			if (sheenWeight > 0) {
				assert(false);
				//disneySheen = DisneySheen(diffuseWeight * sheenWeight * Csheen);
			}
		}

		// Create the microfacet distribution for metallic and/or specular
		// transmission.
		float aspect = sqrt(1 - sd.anisotropic * .9);
		float ax = max(.001f, pow2(rough) / aspect);
		float ay = max(.001f, pow2(rough) * aspect);

		// Specular is Trowbridge-Reitz with a modified Fresnel function.
		float specTint = 0;
		vec3f Cspec0 = any(sd.specular) ? sd.specular : 
			lerp(SchlickR0FromEta(e) * lerp(vec3f(1), Ctint, specTint), c, metallicWeight);

		microfacetBrdf = MicrofacetBrdf(1, e, ax, ay);
		components |= DISNEY_SPEC_REFLECTION;
#if KRR_USE_DISNEY
		microfacetBrdf.disneyR0 = Cspec0;
		microfacetBrdf.metallic = sd.metallic;
#endif
		// Clearcoat
		float cc = 0;
		if (cc > 0) {
			assert(false);
		}

		// specular BTDF if has transmission
		if (strans > 0) {
			//vec3f T = strans * sqrt(c);
			vec3f T = strans;
			if (thin) {
				// Scale roughness based on IOR (Burley 2015, Figure 15).
				assert(false);
				//float rscaled = (0.65f * e - 0.35f) * rough;
				//float ax = max(.001f, pow2(rscaled) / aspect);
				//float ay = max(.001f, pow2(rscaled) * aspect);
				//microfacetBtdf = MicrofacetBtdf(T, 1, e, ax, ay);
			}
			else{
				microfacetBtdf = MicrofacetBtdf(T, 1, e, ax, ay);
			}
			components |= DISNEY_SPEC_TRANSMISSION;
		}

		if (thin) {
			// TODO: add lambertian transmission, weighted by (1 - diffTrans)
			assert(false);
		}

		// calculate sampling weights
		float approxFresnel = luminance(DisneyFresnel(Cspec0, sd.metallic, e, AbsCosTheta(sd.wo)));
		pDiffuse = components & DISNEY_DIFFUSE ? luminance(sd.diffuse) * (1 - sd.metallic) * (1 - sd.specularTransmission) : 0;
		pSpecRefl = components & DISNEY_SPEC_REFLECTION ? luminance(lerp(sd.specular, vec3f(1), approxFresnel)) : 0;
	//	pSpecTrans = components & DISNEY_SPEC_TRANSMISSION ? (1.f - approxFresnel) * (1 - sd.metallic) * sd.specularTransmission * luminance(sd.diffuse) : 0;
		pSpecTrans = components & DISNEY_SPEC_TRANSMISSION ? (1.f - approxFresnel) * (1 - sd.metallic) * sd.specularTransmission: 0;
		float totalWt = pDiffuse + pSpecRefl + pSpecTrans;
		if (totalWt > 0) pDiffuse /= totalWt, pSpecRefl /= totalWt, pSpecTrans /= totalWt;
		//pDiffuse = 0.6, pSpecRefl = 0.4, pSpecTrans = 0;
		//printf("pdiffuse: %f pspecreflect: %f pspectransmit: %f metallic: %f\n",
		//	pDiffuse, pSpecRefl, pSpecTrans, metallicWeight);
	}

	KRR_CALLABLE vec3f f(vec3f wo, vec3f wi) const {
		vec3f val = 0;
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && reflect) {
			if (components & DISNEY_DIFFUSE)val += disneyDiffuse.f(wo, wi);
			if (components & DISNEY_RETRO) val += disneyRetro.f(wo, wi);
		}
		if (pSpecRefl > 0 && (components & DISNEY_SPEC_REFLECTION) && reflect) {
			val += microfacetBrdf.f(wo, wi);
		}
		if (pSpecTrans > 0 && (components & DISNEY_SPEC_TRANSMISSION) && !reflect) {
			val += microfacetBtdf.f(wo, wi);
		}
		return val;
	}

	KRR_CALLABLE BSDFSample sample(vec3f wo, Sampler& sg) const {
		BSDFSample sample;
		float comp = sg.get1D();
		if (comp < pDiffuse) {
			vec3f wi = cosineSampleHemisphere(sg.get2D());
			if (wo.z < 0) wi.z *= -1;
			sample.pdf = pdf(wo, wi);
			sample.f = f(wo, wi);
			sample.wi = wi;
		}
		else if (comp < pDiffuse + pSpecRefl) {
			sample = microfacetBrdf.sample(wo, sg);
			sample.pdf *= pSpecRefl;
			if (pDiffuse) {
				if (components & DISNEY_DIFFUSE)
					sample.f += disneyDiffuse.f(wo, sample.wi);
				if (components & DISNEY_RETRO)
					sample.f += disneyRetro.f(wo, sample.wi);
				sample.pdf += pDiffuse * AbsCosTheta(sample.wi) * M_1_PI;
			}
		}
		else if (pSpecTrans > 0) {
			sample = microfacetBtdf.sample(wo, sg);
			sample.pdf *= pSpecTrans;
		}
		return sample;
	}

	KRR_CALLABLE float pdf(vec3f wo, vec3f wi) const {
		float val = 0;
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && (components & (DISNEY_DIFFUSE | DISNEY_RETRO)) && reflect) {
			val += pDiffuse * AbsCosTheta(wi) * M_1_PI;
		}
		if (pSpecRefl > 0 && (components & DISNEY_SPEC_REFLECTION) && reflect) {
			val += pSpecRefl * microfacetBrdf.pdf(wo, wi);
		}
		if (pSpecTrans > 0 && (components & DISNEY_SPEC_TRANSMISSION) && !reflect) {
			val += pSpecTrans * microfacetBtdf.pdf(wo, wi);
		}
		return val;
	}

	KRR_CALLABLE bool hasComponent(int comp) { return comp & components; }

	//DisneySheen disneySheen;			// [optional] if sheen weight eligible
	//DisneyFakeSS disneyFakeSS;		// [optional] fake ss on thin surface
	DisneyRetro disneyRetro;			// [optional] if diffuse weight eligible
	DisneyDiffuse disneyDiffuse;		// [optional] if diffuse weight eligible
	MicrofacetBrdf microfacetBrdf;		// [always]	 
	MicrofacetBtdf microfacetBtdf;		// [optional] if has specular transmission

	int components{ 0 };

	float pDiffuse{ 0 };
	float pSpecTrans{ 0 };
	float pSpecRefl{ 0 };

#if 0
	vec3f color{ 1 };
	float metallic{ 0 };
	float eta{ 1.5 };
	float roughness{ 1 };
	float specularTint{ 0 };
	float anisotropic{ 0 };
	float sheen{ 0 };
	float sheenTint{ 0 };
	float clearcoat{ 0 };
	float clearcoatGloss{ 0 };
	float specTrans{ 0 };
	float scatterDistance{ 0 };
	bool thin{ 0 };
	float flatness{ 0 };
	float diffuseTrans{ 0 };
	float bumpMap{ 0 };
#endif
};

KRR_NAMESPACE_END