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

	KRR_CALLABLE DisneyDiffuse(const Color &R) : R(R) {}
	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		if (!SameHemisphere(wo, wi)) return 0;
		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));
		return R * M_INV_PI * (1 - Fo / 2) * (1 - Fi / 2);
	}
	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector3f *, const Vector3f *) const { return R; }

	Color R;
};

class DisneyFakeSS{
public:
	DisneyFakeSS() = default;

	KRR_CALLABLE DisneyFakeSS(const Color &R, float roughness)
		: R(R), roughness(roughness) {}

	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		Vector3f wh = wi + wo;
		if (!wh.any()) return Vector3f(0);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		// Fss90 used to "flatten" retroreflection based on roughness
		float Fss90 = cosThetaD * cosThetaD * roughness;
		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));
		float Fss = lerp(1.f, Fss90, Fo) * lerp(1.f, Fss90, Fi);
		// 1.25 scale is used to (roughly) preserve albedo
		float ss = 1.25f * (Fss * (1 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - .5f) + .5f);

		return R * M_INV_PI * ss;
	};

	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector2f *, const Vector2f *) const { return R; }

	Color R;
	float roughness;
};

class DisneyRetro{
public:
	DisneyRetro() = default;
	KRR_CALLABLE DisneyRetro(const Color& R, float roughness)
		: R(R), roughness(roughness) {}
	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		Vector3f wh = wi + wo;
		if (wh[0] == 0 && wh[1] == 0 && wh[2] == 0) return Vector3f(0.);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		float Fo = SchlickWeight(AbsCosTheta(wo)),
			Fi = SchlickWeight(AbsCosTheta(wi));
		float Rr = 2 * roughness * cosThetaD * cosThetaD;

		// Burley 2015, eq (4).
		return R * M_INV_PI * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
	};
	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector2f *, const Vector2f *) const { return R; }

	Color R;
	float roughness;
};

class DisneySheen{
public:
	DisneySheen() = default;
	KRR_CALLABLE DisneySheen(const Vector3f& R): R(R) {}
	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		Vector3f wh = wi + wo;
		if (!wh.any()) return Vector3f(0.);
		wh = normalize(wh);
		float cosThetaD = dot(wi, wh);

		return R * SchlickWeight(cosThetaD);
	}

	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector2f *, const Vector2f *) const { return R; }

	Color R;
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
	//return 2 / (1 + sqrt(1 + alpha2 * (1 - cosTheta2) / cosTheta2));
}

class DisneyClearcoat{
public:
	DisneyClearcoat() = default;

	KRR_CALLABLE DisneyClearcoat(float weight, float gloss)
		: weight(weight), gloss(gloss) {}

	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		Vector3f wh = wi + wo;
		if (!wh.any()) return Vector3f(0);
		wh = normalize(wh);

		float Dr = GTR1(AbsCosTheta(wh), gloss);
		float Fr = FrSchlick(.04f, 1.f, dot(wo, wh));
		// The geometric term always based on alpha = 0.25.
		float Gr = smithG_GGX(AbsCosTheta(wo), .25) * smithG_GGX(AbsCosTheta(wi), .25);

		return weight * Gr * Fr * Dr / 4;
	};

	KRR_CALLABLE Color Sample_f(const Vector3f &wo, Vector3f *wi, const Vector2f &u,
		float* pdf, BSDFType* sampledType) const {

		if (wo[2] == 0) return Color::Zero();

		float alpha2 = gloss * gloss;
		float cosTheta = sqrt(max(float(0), (1 - pow(alpha2, 1 - u[0])) / (1 - alpha2)));
		float sinTheta = sqrt(max((float)0, 1 - cosTheta * cosTheta));
		float phi = 2 * M_PI * u[1];
		Vector3f wh = sphericalToCartesian(sinTheta, cosTheta, phi);
		if (!SameHemisphere(wo, wh)) wh = -wh;

		*wi = Reflect(wo, wh);
		if (!SameHemisphere(wo, *wi)) return Color(0);

		*pdf = Pdf(wo, *wi);
		return f(wo, *wi);
	};

	KRR_CALLABLE float Pdf(const Vector3f& wo, const Vector3f& wi) const {
		if (!SameHemisphere(wo, wi)) return 0;

		Vector3f wh = wi + wo;
		if (wh[0] == 0 && wh[1] == 0 && wh[2] == 0) return 0;
		wh = normalize(wh);

		float Dr = GTR1(AbsCosTheta(wh), gloss);
		return Dr * AbsCosTheta(wh) / (4 * dot(wo, wh));
	};

	float weight, gloss;
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
	/* Thin surface is not supported. */
	/* The clearcoat lobe is not added, since its optional and controlled by an additional intensity. */
public:
	DisneyBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(DisneyBsdf);

	KRR_CALLABLE void setup(const ShadingData& sd) {

		Color c				 = sd.diffuse;
		float metallicWeight = sd.metallic;
		float e				 = sd.IoR;
		float strans		 = sd.specularTransmission;
		float diffuseWeight	 = (1 - metallicWeight) * (1 - strans);
		float dt			 = sd.diffuseTransmission;
		float roughness		 = sd.roughness;
		float lum			 = luminance(c);
		// normalize lum. to isolate hue+sat
		Color Ctint = Color::Ones();
		if (lum > 0) Ctint = c / lum;

		float sheenWeight = 0;
		Color Csheen;
		if (sheenWeight > 0) {	//unused
			float stint = 0;
			Csheen		= lerp(Color(1), Ctint, stint);
		}

		if (diffuseWeight > 0) {

			// No subsurface scattering; use regular (Fresnel modified) diffuse.
			disneyDiffuse = DisneyDiffuse(diffuseWeight * c);
			components |= DISNEY_DIFFUSE;
			
			disneyRetro = DisneyRetro(diffuseWeight * c, roughness);
			components |= DISNEY_RETRO;

			if (sheenWeight > 0) {	// unused
				//disneySheen = DisneySheen(diffuseWeight * sheenWeight * Csheen);
			}
		}

		// Create the microfacet distribution for metallic and/or specular
		// transmission.
		float aspect = sqrt(1 - sd.anisotropic * .9);
		float ax	 = max(.001f, pow2(roughness) / aspect);
		float ay	 = max(.001f, pow2(roughness) * aspect);

		// Specular is Trowbridge-Reitz with a modified Fresnel function.
		float specTint = 1;		// [unused] this is manually set to 1...
		Color Cspec0   = sd.specular;
		if (!any(sd.specular)) 
			Cspec0 = lerp(SchlickR0FromEta(e) * lerp(Color::Ones(), Ctint, specTint), c, metallicWeight);

		microfacetBrdf = MicrofacetBrdf(Color::Ones(), e, ax, ay);
		components |= DISNEY_SPEC_REFLECTION;
#if KRR_USE_DISNEY
		microfacetBrdf.disneyR0 = Cspec0;
		microfacetBrdf.metallic = metallicWeight;
#endif

		// specular BTDF if has transmission
		if (strans > 0) {
			Color T = strans * sqrt(c);

			microfacetBtdf = MicrofacetBtdf(T, 1, e, ax, ay);
#if KRR_USE_DISNEY
			microfacetBtdf.disneyR0 = Cspec0;
			microfacetBtdf.metallic = metallicWeight;
#endif
			components |= DISNEY_SPEC_TRANSMISSION;
		}

		// calculate sampling weights
		float approxFresnel =
			luminance(DisneyFresnel(Cspec0, metallicWeight, e, AbsCosTheta(sd.wo)));
		pDiffuse	  = components & DISNEY_DIFFUSE ? sd.diffuse.mean() * (1 - metallicWeight) *
													  (1 - sd.specularTransmission)
												: 0;
		pSpecRefl	  = components & DISNEY_SPEC_REFLECTION
								? lerp(sd.specular, Color::Ones(), approxFresnel).mean() *
							(1 - sd.specularTransmission * (1 - metallicWeight))
								: 0;
		pSpecTrans	  = components & DISNEY_SPEC_TRANSMISSION
							? (1 - approxFresnel) * (1 - metallicWeight) * sd.specularTransmission
							: 0;
		float totalWt = pDiffuse + pSpecRefl + pSpecTrans;
		if (totalWt > 0) pDiffuse /= totalWt, pSpecRefl /= totalWt, pSpecTrans /= totalWt;
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		Color val	 = Color::Zero();
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && reflect) {
			if (components & DISNEY_DIFFUSE) val += disneyDiffuse.f(wo, wi);
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

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler& sg) const {
		BSDFSample sample;
		float comp = sg.get1D();
		if (comp < pDiffuse) {
			Vector3f wi = cosineSampleHemisphere(sg.get2D());
			if (wo[2] < 0) wi[2] *= -1;
			sample.pdf	 = pdf(wo, wi);
			sample.f	 = f(wo, wi);
			sample.wi	 = wi;
			sample.flags = BSDF_DIFFUSE_REFLECTION;
		} else if (comp < pDiffuse + pSpecRefl) {
			sample = microfacetBrdf.sample(wo, sg);
			sample.pdf *= pSpecRefl;
			if (pDiffuse) {
				if (components & DISNEY_DIFFUSE)
					sample.f += disneyDiffuse.f(wo, sample.wi);
				if (components & DISNEY_RETRO)
					sample.f += disneyRetro.f(wo, sample.wi);
				sample.pdf += pDiffuse * AbsCosTheta(sample.wi) * M_INV_PI;
			}
		} else if (pSpecTrans > 0) {
			sample = microfacetBtdf.sample(wo, sg);
			sample.pdf *= pSpecTrans;
		}
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		float val = 0;
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && (components & (DISNEY_DIFFUSE | DISNEY_RETRO)) && reflect) {
			val += pDiffuse * AbsCosTheta(wi) * M_INV_PI;
		}
		if (pSpecRefl > 0 && (components & DISNEY_SPEC_REFLECTION) && reflect) {
			val += pSpecRefl * microfacetBrdf.pdf(wo, wi);
		}
		if (pSpecTrans > 0 && (components & DISNEY_SPEC_TRANSMISSION) && !reflect) {
			val += pSpecTrans * microfacetBtdf.pdf(wo, wi);
		}
		return val;
	}

	KRR_CALLABLE BSDFType flags() const {
		BSDFType type = pDiffuse > 0 ? BSDF_DIFFUSE_REFLECTION : BSDF_UNSET;
		return type | microfacetBrdf.flags() | microfacetBtdf.flags();
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

};

KRR_NAMESPACE_END