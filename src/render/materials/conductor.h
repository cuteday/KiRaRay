#pragma once
#include "common.h"

NAMESPACE_BEGIN(krr)

class ConductorBsdf {
public:
	ConductorBsdf() = default;
	KRR_CALLABLE ConductorBsdf(Spectrum color, float alphaX, float alphaY) {
		reflectance = color.cwiseMax(0.f).cwiseMin(.9999f);
		eta = Spectrum(1);
		k = 2 * reflectance.sqrt() / (Spectrum::Ones() - reflectance).sqrt();
		distribution = GGXMicrofacetDistribution(alphaX, alphaY);
	}

	_DEFINE_BSDF_INTERNAL_ROUTINES(ConductorBsdf);

	KRR_CALLABLE BSDFEval eval(Vector3f wo, Vector3f wi,
							 TransportMode mode = TransportMode::Radiance) const {
		return {fImpl(wo, wi, mode, distribution), pdfImpl(wo, wi, mode, distribution)};
	}

	KRR_CALLABLE BSDFEval eval(Vector3f wo, Vector3f wi, TransportMode mode, MicrofacetType type) const {
		if (type == MicrofacetType::Beckmann) {
			auto d = beckmann();
			return {fImpl(wo, wi, mode, d), pdfImpl(wo, wi, mode, d)};
		}
		return eval(wo, wi, mode);
	}

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
		TransportMode mode = TransportMode::Radiance) const {
		return fImpl(wo, wi, mode, distribution);
	}

	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi, TransportMode mode, MicrofacetType type) const {
		if (type == MicrofacetType::Beckmann) return fImpl(wo, wi, mode, beckmann());
		return fImpl(wo, wi, mode, distribution);
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
		TransportMode mode = TransportMode::Radiance) const {
		return pdfImpl(wo, wi, mode, distribution);
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi, TransportMode mode, MicrofacetType type) const {
		if (type == MicrofacetType::Beckmann) return pdfImpl(wo, wi, mode, beckmann());
		return pdfImpl(wo, wi, mode, distribution);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
		TransportMode mode = TransportMode::Radiance) const {
		return sampleImpl(wo, sg, mode, distribution);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg, TransportMode mode, MicrofacetType type) const {
		if (type == MicrofacetType::Beckmann) return sampleImpl(wo, sg, mode, beckmann());
		return sampleImpl(wo, sg, mode, distribution);
	}

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) {
		Spectra eta_spec = intr.material->mMaterialParams.spectralEta, k_spec;
		reflectance		 = intr.sd.diffuse.cwiseMin(0.9999); // avoid NaN caused by r==1... 
		float aspect	 = sqrt(1 - intr.sd.anisotropic * .9f);
		float ax		 = max(.001f, pow2(intr.sd.roughness) / aspect);
		float ay		 = max(.001f, pow2(intr.sd.roughness) * aspect);
		distribution	 = GGXMicrofacetDistribution{ax, ay};
		// currently only the spectral build supports wavelength-dependent eta and k...
#if KRR_RENDER_SPECTRAL
		if (eta_spec) {
			eta = intr.material->mMaterialParams.spectralEta.sample(intr.lambda);
		} else 
#endif
			eta = Spectrum::Constant(intr.sd.IoR);
#if KRR_RENDER_SPECTRAL
		if (k_spec) {
			k = intr.material->mMaterialParams.spectralK.sample(intr.lambda);
		} else 
#endif
			k = 2 * reflectance.sqrt() / (Spectrum::Ones() - reflectance).cwiseMax(0).sqrt();
	}

	KRR_CALLABLE BSDFType flags() const {
		return BSDF_REFLECTION | (distribution.isDelta() ? BSDF_SPECULAR : BSDF_GLOSSY);
	}

private:
	template<class Distribution>
	KRR_CALLABLE Spectrum fImpl(Vector3f wo, Vector3f wi,
							TransportMode mode, const Distribution &distribution) const {
		if (!SameHemisphere(wo, wi)) return Spectrum::Zero();
		if (distribution.isDelta()) return Spectrum::Zero();
		// Evaluate rough conductor BRDF
		// Compute cosines and $\wm$ for conductor BRDF
		float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
		if (cosTheta_i == 0 || cosTheta_o == 0) return Spectrum::Zero();
		Vector3f wm = wi + wo;
		if (wm.squaredNorm() == 0) return Spectrum::Zero();
		wm.normalize();

		// Evaluate Fresnel factor _F_ for conductor BRDF
		Spectrum F = FrComplex(AbsDot(wo, wm), eta, k);

		return distribution.D(wm) * F * distribution.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
	}

	template<class Distribution>
	KRR_CALLABLE BSDFSample sampleImpl(Vector3f wo, Sampler &sg,
								   TransportMode mode, const Distribution &distribution) const {
		if (distribution.isDelta()) {
			// Sample perfect specular conductor BRDF
			Vector3f wi(-wo.x(), -wo.y(), wo.z());
			Spectrum f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
			return BSDFSample(f, wi, 1, BSDFType::BSDF_SPECULAR_REFLECTION);
		}
		// Sample rough conductor BRDF
		// Sample microfacet normal $\wm$ and reflected direction $\wi$
		if (wo.z() == 0) return {};
		Vector3f wm = distribution.Sample(wo, sg.get2D());
		Vector3f wi = Reflect(wo, wm);
		if (!SameHemisphere(wo, wi)) return {};

		// Compute PDF of _wi_ for microfacet reflection
		float pdf = distribution.Pdf(wo, wm) / (4 * AbsDot(wo, wm));

		float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
		if (cosTheta_i == 0 || cosTheta_o == 0) return {};
		// Evaluate Fresnel factor _F_ for conductor BRDF
		Spectrum F = FrComplex(AbsDot(wo, wm), eta, k);

		Spectrum f =
			distribution.D(wm) * F * distribution.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
		return BSDFSample(f, wi, pdf, BSDFType::BSDF_GLOSSY_REFLECTION);
	}

	template<class Distribution>
	KRR_CALLABLE float pdfImpl(Vector3f wo, Vector3f wi,
						   TransportMode mode, const Distribution &distribution) const {
		if (!SameHemisphere(wo, wi)) return 0;
		if (distribution.isDelta()) return 0;
		// Evaluate sampling PDF of rough conductor BRDF
		Vector3f wm = wo + wi;
		if (wm.isZero()) return 0;
		wm = FaceForward(wm.normalized(), Vector3f::UnitZ());
		return distribution.Pdf(wo, wm) / (4 * AbsDot(wo, wm));
	}

	KRR_CALLABLE BeckmannMicrofacetDistribution beckmann() const {
		Vector2f alpha = distribution.alpha();
		return {alpha[0], alpha[1]};
	}

	Spectrum eta, k;
	Spectrum reflectance;
	GGXMicrofacetDistribution distribution;
};

NAMESPACE_END(krr)
