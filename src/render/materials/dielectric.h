#pragma once

#include "common.h"

#include "util/math_utils.h"
#include "sampler.h"

#include "bxdf.h"
#include "microfacet.h"
#include "matutils.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace bsdf;


class DielectricBsdf {
public:
	DielectricBsdf() = default;
	
	_DEFINE_BSDF_INTERNAL_ROUTINES(DielectricBsdf);

	KRR_CALLABLE DielectricBsdf(Spectrum base, float eta, float alpha_x, float alpha_y) {
		baseColor	 = base;
		eta			 = eta;
		distribution = GGXMicrofacetDistribution(alpha_x, alpha_y);
	}

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) { 
		baseColor	 = intr.sd.diffuse * intr.sd.specularTransmission;
		eta			 = intr.sd.IoR;
		float alpha	 = pow2(intr.sd.roughness);
		distribution = { alpha, alpha };
	}

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
								   TransportMode mode = TransportMode::Radiance) const {
		if (eta == 1 || distribution.isSpecular()) return Spectrum::Zero();
		// Evaluate rough dielectric BSDF
		// Compute generalized half vector _wm_
		float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
		bool reflect = cosTheta_i * cosTheta_o > 0;
		float etap	 = 1;
		if (!reflect)
			etap = cosTheta_o > 0 ? eta : (1 / eta);
		Vector3f wm = wi * etap + wo;
		if (cosTheta_i == 0 || cosTheta_o == 0 || !wm.any()) return Spectrum::Zero();
		wm = FaceForward(normalize(wm), Vector3f(0, 0, 1));

		// Discard backfacing microfacets
		if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
			return Spectrum::Zero();

		float F	  = FrDielectric(copysignf(dot(wo, wm), wo[2]), eta);
		Spectrum R = baseColor * F, T = baseColor * (1 - F);
		if (reflect) {
			// Compute reflection at rough dielectric interface
			return Spectrum(distribution.D(wm) * distribution.G(wo, wi) * R /
								   abs(4 * cosTheta_i * cosTheta_o));

		} else {
			// Compute transmission at rough dielectric interface
			float denom = pow2(dot(wi, wm) + dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
			Spectrum ft	= T * distribution.D(wm) * distribution.G(wo, wi) *
					   abs(dot(wi, wm) * dot(wo, wm) / denom);
			// Account for non-symmetry with transmission to different medium
			if (mode == TransportMode::Radiance) ft /= pow2(etap); 
			return ft;
		}
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		if (eta == 1 || distribution.isSpecular()) {
			// Sample perfectly specular dielectric BSDF
			float F = FrDielectric(CosTheta(wo), eta);
			Spectrum R = baseColor * F, T = baseColor * (1 - F);
			// Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
			float pr = R.mean(), pt = T.mean();
			if (pr == 0 && pt == 0)
				return {};

			if (sg.get1D() < pr / (pr + pt)) {
				// Sample perfect specular dielectric BRDF
				Vector3f wi(-wo[0], -wo[1], wo[2]);
				Spectrum fr(R / AbsCosTheta(wi));
				return BSDFSample(fr, wi, pr / (pr + pt), BSDF_SPECULAR_REFLECTION);

			} else {
				// Sample perfect specular dielectric BTDF
				// Compute ray direction for specular transmission
				Vector3f wi;
				float etap;
				bool valid = Refract(wo, Vector3f(0, 0, copysignf(1, wo[2])), eta, &etap, &wi);
				if (!valid)
					return {};

				Spectrum ft(T / AbsCosTheta(wi));
				// Account for non-symmetry with transmission to different medium
				if (mode == TransportMode::Radiance) ft /= pow2(etap);

				return BSDFSample(ft, wi, pt / (pr + pt), BSDF_SPECULAR_TRANSMISSION);
			}

		} else {
			// Sample rough dielectric BSDF
			Vector3f wm = distribution.Sample(wo, sg.get2D());
			float F		= FrDielectric(copysignf(dot(wo, wm), wo[2]), eta);
			Spectrum R = baseColor * F, T = baseColor * (1 - F);
			// Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
			float pr = R.mean(), pt = T.mean();
			if (pr == 0 && pt == 0)
				return {};

			float pdf;
			if (sg.get1D() < pr / (pr + pt)) {
				// Sample reflection at rough dielectric interface
				Vector3f wi = Reflect(wo, wm);
				if (!SameHemisphere(wo, wi)) return {};
				// Compute PDF of rough dielectric reflection
				pdf = distribution.Pdf(wo, wm) / (4 * fabs(dot(wo, wm))) * pr / (pr + pt);

				Spectrum f = distribution.D(wm) * distribution.G(wo, wi) * R /
									(4 * CosTheta(wi) * CosTheta(wo));
				return BSDFSample(f, wi, pdf, BSDF_GLOSSY_REFLECTION);

			} else {
				// Sample transmission at rough dielectric interface
				float etap;
				Vector3f wi;
				bool tir = !Refract(wo, (Vector3f) wm, eta, &etap, &wi);
				if (SameHemisphere(wo, wi) || wi[2] == 0 || tir)
					return {};
				// Compute PDF of rough dielectric transmission
				float denom	  = pow2(dot(wi, wm) + dot(wo, wm) / etap);
				float dwm_dwi = fabs(dot(wi, wm)) / denom;
				pdf			  = distribution.Pdf(wo, wm) * dwm_dwi * pt / (pr + pt);

				// Evaluate BRDF and return _BSDFSample_ for rough transmission
				Spectrum ft(T * distribution.D(wm) * distribution.G(wo, wi) *
					abs(dot(wi, wm) * dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
				// Account for non-symmetry with transmission to different medium
				if (mode == TransportMode::Radiance) ft /= pow2(etap);

				return BSDFSample(ft, wi, pdf, BSDF_GLOSSY_TRANSMISSION);
			}
		}
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		if (eta == 1 || distribution.isSpecular())
			return 0;
		// Evaluate sampling PDF of rough dielectric BSDF
		// Compute generalized half vector _wm_
		float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
		bool reflect = cosTheta_i * cosTheta_o > 0;
		float etap	 = 1;
		if (!reflect)
			etap = cosTheta_o > 0 ? eta : (1 / eta);
		Vector3f wm = wi * etap + wo;
		if (cosTheta_i == 0 || cosTheta_o == 0 || wm.squaredNorm() == 0)
			return {};
		wm = FaceForward(normalize(wm), Vector3f(0, 0, 1));

		// Discard backfacing microfacets
		if (dot(wm, wi) * cosTheta_i < 0 || dot(wm, wo) * cosTheta_o < 0)
			return {};

		// Determine Fresnel reflectance of rough dielectric boundary
		float F	  = FrDielectric(CosTheta(wo), eta);
		Spectrum R = baseColor * F, T = baseColor * (1 - F);

		// Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
		float pr = R.mean(), pt = T.mean();
		if (pr == 0 && pt == 0)
			return {};

		// Return PDF for rough dielectric
		float pdf;
		if (reflect) {
			// Compute PDF of rough dielectric reflection
			pdf = distribution.Pdf(wo, wm) / (4 * fabs(dot(wo, wm))) * pr / (pr + pt);

		} else {
			// Compute PDF of rough dielectric transmission
			float denom	  = pow2(dot(wi, wm) + dot(wo, wm) / etap);
			float dwm_dwi = fabs(dot(wi, wm)) / denom;
			pdf			  = distribution.Pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
		}
		return pdf;
	}

	KRR_CALLABLE BSDFType flags() const { 
		BSDFType type = eta == 1 ? BSDF_TRANSMISSION : (BSDF_REFLECTION | BSDF_TRANSMISSION);
		return type | (distribution.isSpecular() ? BSDF_SPECULAR : BSDF_GLOSSY);
	}

private:
	float eta;	
	Spectrum baseColor;
	GGXMicrofacetDistribution distribution;
};

KRR_NAMESPACE_END