#pragma once

#include "common.h"
#include "math/math.h"
#include "math/utils.h"
#include "sampler.h"

#include "bxdf.h"
#include "disney.h"
#include "microfacet.h"
#include "dielectric.h"
#include "matutils.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;
using namespace shader;

class PrincipledDiffuse {
public:
	PrincipledDiffuse() = default;

	KRR_CALLABLE PrincipledDiffuse(const Color &R) : R(R) {}
	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		if (!SameHemisphere(wo, wi))
			return 0;
		float Fo = SchlickWeight(AbsCosTheta(wo)), Fi = SchlickWeight(AbsCosTheta(wi));
		return R * M_INV_PI * (1 - Fo / 2) * (1 - Fi / 2);
	}
	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector3f *, const Vector3f *) const { return R; }

	Color R;
};

class PrincipledRetro {
public:
	PrincipledRetro() = default;
	KRR_CALLABLE PrincipledRetro(const Color &R, float roughness) : R(R), roughness(roughness) {}
	KRR_CALLABLE Color f(const Vector3f &wo, const Vector3f &wi) const {
		Vector3f wh = wi + wo;
		if (wh[0] == 0 && wh[1] == 0 && wh[2] == 0)
			return Vector3f(0.);
		wh				= normalize(wh);
		float cosThetaD = dot(wi, wh);

		float Fo = SchlickWeight(AbsCosTheta(wo)), Fi = SchlickWeight(AbsCosTheta(wi));
		float Rr = 2 * roughness * cosThetaD * cosThetaD;

		// Burley 2015, eq (4).
		return R * M_INV_PI * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
	};
	KRR_CALLABLE Color rho(const Vector3f &, int, const Vector2f *) const { return R; }
	KRR_CALLABLE Color rho(int, const Vector2f *, const Vector2f *) const { return R; }

	Color R;
	float roughness;
};

class PrincipledBsdf {
public:
	enum PrincipledComponent {
		PRINCIPLED_DIFFUSE = 1 << 1,
		PRINCIPLED_METAL   = 1 << 2,
		PRINCIPLED_GLASS   = 1 << 3
	};

	_DEFINE_BSDF_INTERNAL_ROUTINES(PrincipledBsdf);

	KRR_CALLABLE void setup(const ShadingData &sd) { 
		Color c				 = sd.diffuse;		/* base color*/
		float metallicWeight = sd.metallic;
		float e				 = sd.IoR;
		float strans		 = sd.specularTransmission;
		float diffuseWeight	 = (1 - metallicWeight) * (1 - strans);
		float dt			 = sd.diffuseTransmission;
		float roughness		 = sd.roughness;
		float lum			 = luminance(c);
		// normalize lum. to isolate hue+sat
		Color Ctint = Color::Ones();
		if (lum > 0)
			Ctint = c / lum;

		float sheenWeight = 0;
		Color Csheen;
		if (sheenWeight > 0) { // unused
			float stint = 0;
			Csheen		= lerp(Color(1), Ctint, stint);
		}

		if (diffuseWeight > 0) {
			// No subsurface scattering; use regular (Fresnel modified) diffuse.
			components |= PRINCIPLED_DIFFUSE;
			disneyDiffuse = PrincipledDiffuse(c);
			disneyRetro = PrincipledRetro(c, roughness);
		}

		// Create the microfacet distribution for metallic and/or specular
		// transmission.
		float aspect = sqrt(1 - sd.anisotropic * .9);
		float ax	 = max(1e-3f, pow2(roughness) / aspect);
		float ay	 = max(1e-3f, pow2(roughness) * aspect);
		delta		 = max(ax, ay) <= 1e-3f;

		// Specular is Trowbridge-Reitz with a modified Fresnel function.
		Color Cspec0   = sd.specular;
		if (!any(sd.specular))
			Cspec0 =
				lerp(SchlickR0FromEta(e) * lerp(Color::Ones(), Ctint, 1), c, metallicWeight);

		metalBrdf = MicrofacetBrdf(Color(1), e, ax, ay);
		components |= PRINCIPLED_METAL;
#if KRR_USE_DISNEY
		metalBrdf.disneyR0 = Cspec0;
		metalBrdf.metallic = metallicWeight;
#endif

		if (strans > 0) {
			glassBsdf = DielectricBsdf(c, 1/e, ax, ay);
			components |= PRINCIPLED_GLASS;
		}
		
		/* Sampling weights */
		weightDiffuse	  = components & PRINCIPLED_DIFFUSE
							? (1 - metallicWeight) * (1 - strans)
							: 0;
		weightMetal	  = components & PRINCIPLED_METAL ? (1 - strans * (1 - metallicWeight))
							: 0;
		weightGlass	  = components & PRINCIPLED_GLASS ? (1 - metallicWeight) * strans
							: 0;
		float totalWt = weightDiffuse + weightMetal + weightGlass;
		if (totalWt > 0)
			pDiffuse = weightDiffuse / totalWt, 
			pMetal = weightMetal / totalWt, 
			pGlass = weightGlass / totalWt;
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi) const {
		Color val	 = Color::Zero();
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && reflect) {
			val += weightDiffuse * disneyDiffuse.f(wo, wi);
			val += weightDiffuse * disneyRetro.f(wo, wi);
		}
		if (pMetal > 0 && reflect) {
			val += weightMetal * metalBrdf.f(wo, wi);
		}
		if (pGlass > 0) {
			val += weightGlass * glassBsdf.f(wo, wi);
		}
		return val;
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg) const {
		BSDFSample sample{};
		float comp = sg.get1D();
		if (comp < pDiffuse) {
			Vector3f wi = cosineSampleHemisphere(sg.get2D());
			if (wo[2] < 0)
				wi[2] *= -1;
			sample.pdf	 = pdf(wo, wi);
			sample.f	 = f(wo, wi);
			sample.wi	 = wi;
			sample.flags = BSDF_DIFFUSE_REFLECTION;
		} else if (comp < pDiffuse + pMetal) {
			sample = metalBrdf.sample(wo, sg);
			sample.pdf *= pMetal;
			if (pDiffuse > 0) {
				sample.f += weightDiffuse * disneyDiffuse.f(wo, sample.wi);
				sample.f += weightDiffuse * disneyRetro.f(wo, sample.wi);
				sample.pdf += pDiffuse * AbsCosTheta(sample.wi) * M_INV_PI;
			}
			if (pGlass > 0) {
				sample.f += weightGlass * glassBsdf.f(wo, sample.wi);
				sample.pdf += pGlass * glassBsdf.pdf(wo, sample.wi);
			}
		} else if (pGlass > 0) {
			sample = glassBsdf.sample(wo, sg);
			if (!sample.f.any() || sample.pdf == 0)
				return {};
			sample.pdf *= pGlass;
			if (SameHemisphere(wo, sample.wi)) {
				if (pDiffuse > 0) {
					sample.f += weightDiffuse * disneyDiffuse.f(wo, sample.wi);
					sample.f += weightDiffuse * disneyRetro.f(wo, sample.wi);
					sample.pdf += pDiffuse * AbsCosTheta(sample.wi) * M_INV_PI;
				}
				if (pMetal > 0) {
					sample.f += weightMetal * metalBrdf.f(wo, sample.wi);
					sample.pdf += pMetal * metalBrdf.pdf(wo, sample.wi);
				}
			}
		}
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi) const {
		float val	 = 0;
		bool reflect = SameHemisphere(wo, wi);
		if (pDiffuse > 0 && reflect) {
			val += pDiffuse * AbsCosTheta(wi) * M_INV_PI;
		}
		if (pMetal > 0 && reflect) {
			val += pMetal * metalBrdf.pdf(wo, wi);
		}
		if (pGlass > 0) {
			val += pGlass * glassBsdf.pdf(wo, wi);
		}
		return val;
	}

	KRR_CALLABLE BSDFType flags() const { 
		BSDFType type = pDiffuse > 0 ? BSDF_DIFFUSE_REFLECTION : BSDF_UNSET;
		return type | metalBrdf.flags() | glassBsdf.flags();
	}

private:
	PrincipledRetro disneyRetro;		// [optional] if diffuse weight eligible
	PrincipledDiffuse disneyDiffuse;	// [optional] if diffuse weight eligible
	MicrofacetBrdf metalBrdf;			// [always]
	DielectricBsdf glassBsdf;			// [optional] if has specular transmission

	int components{ 0 };
	bool delta{ 0 };

	float weightDiffuse{}, weightMetal{}, weightGlass{};
	float pDiffuse{ 0 }, pMetal{ 0 }, pGlass{ 0 };
};

KRR_NAMESPACE_END