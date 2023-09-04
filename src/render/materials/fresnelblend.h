#pragma once

#include "common.h"
#include "taggedptr.h"


#include "util/math_utils.h"

#include "bxdf.h"
#include "matutils.h"
#include "fresnel.h"
#include "microfacet.h"
#include "render/sampling.h"
#include "sampler.h"

KRR_NAMESPACE_BEGIN
using namespace shader;

class FresnelBlendBrdf {
public:
	FresnelBlendBrdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(FresnelBlendBrdf);

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) {
		diffuse		= intr.sd.diffuse;
		specular	= intr.sd.specular;
		float alpha = ggx.RoughnessToAlpha(intr.sd.roughness);
		alpha		= intr.sd.roughness * intr.sd.roughness;
		ggx			= { alpha, alpha };
	}

	KRR_CALLABLE Color f(Vector3f wo, Vector3f wi,
						 TransportMode mode = TransportMode::Radiance) const {
		if (!SameHemisphere(wo, wi))
			return Color::Zero();
		Color diff = (28.f / (23.f * M_PI)) * diffuse * (Color::Ones() - specular)
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wi)))
			* (1.f - pow5(1.f - 0.5f * AbsCosTheta(wo)));
		Vector3f wh = wi + wo;
		if (!any(wh)) return diff;
		wh = normalize(wh);
		float D = ggx.D(wh);
		Color F	   = FrSchlick(specular, Vector3f(1.f), dot(wi, wh));
		Color spec = D * F * 0.25f / (fabs(dot(wi, wh)) * max(AbsCosTheta(wi), AbsCosTheta(wo)));

		return diff + spec;
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		BSDFSample sample = {};
		float comp = sg.get1D();
		Vector2f u = sg.get2D();
		Vector3f wi;
		if (comp < 0.5f) {
			wi = cosineSampleHemisphere(u);
			wi = ToSameHemisphere(wi, wo);
		}
		else {
			Vector3f wh = ggx.Sample(wo, u);
			wi = Reflect(wo, wh);
			if (!SameHemisphere(wi, wo)) return sample;
		}
		sample.wi = wi;
		sample.f = f(wo, sample.wi);
		sample.pdf = pdf(wo, sample.wi);
		return sample;
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		if (!SameHemisphere(wo, wi)) return 0;
		float diffPdf = fabs(wi[2]) * M_INV_PI;

		Vector3f wh = normalize(wo + wi);
		float specPdf = ggx.Pdf(wo, wh) / (4 * dot(wo, wh));

		return 0.5f * (diffPdf + specPdf);
	}

	KRR_CALLABLE BSDFType flags() const {
		BSDFType type = diffuse.any() ? BSDF_DIFFUSE_REFLECTION : BSDF_UNSET;
		if (specular.any())
			type = type | (ggx.isSpecular() ? BSDF_SPECULAR_REFLECTION : BSDF_GLOSSY_REFLECTION);
		return type;
	}

private:
	Color diffuse;
	Color specular;
	GGXMicrofacetDistribution ggx;
};

KRR_NAMESPACE_END