#pragma once

#include "common.h"
#include "taggedptr.h"

#include "math/math.h"
#include "math/utils.h"

#include "bxdf.h"
#include "matutils.h"
#include "diffuse.h"
#include "microfacet.h"
#include "fresnel.h"
#include "render/sampling.h"
#include "sampler.h"


KRR_NAMESPACE_BEGIN
using namespace shader;

class UniformBsdf {
	// A BSDF model taken from NVIDIA's render framework Falcor.
	// Hmm... seems not energy conservative!
public:
	UniformBsdf() = default;

	_DEFINE_BSDF_INTERNAL_ROUTINES(UniformBsdf);

	__both__ void setup(const ShadingData & sd) {
		specTrans = sd.specularTransmission;

		diffuseBsdf.setup(sd);
		microfacetBrdf.setup(sd);
		microfacetBtdf.setup(sd);

		float metallic = sd.metallic;

		float metallicBRDF = metallic;
		float specularBSDF = (1 - metallic) * specTrans;
		float dielectricBSDF = (1 - metallic) * (1 - specTrans);

		float diffuseWeight = luminance(sd.diffuse);
		float specularWeight = FrSchlick(luminance(sd.specular), 1.f, dot(sd.wo, sd.frame.N));
	
		pDiff = diffuseWeight * dielectricBSDF;
		pSpecRefl = specularWeight * (metallicBRDF + dielectricBSDF);
		pSpecTrans = specTrans > 0 ? specularBSDF : 0;

		float norm = pDiff + pSpecRefl + pSpecTrans;
		if (norm > 0) pDiff /= norm, pSpecRefl /= norm, pSpecTrans /= norm;
	}

	__both__ Color f(Vec3f wo, Vec3f wi) const {
		Color val = Color::Zero();
		if(pDiff) val += (1 - specTrans) * diffuseBsdf.f(wo, wi);
		if(pSpecRefl) val += (1 - specTrans) * microfacetBrdf.f(wo, wi);
		if(pSpecTrans) val += specTrans * microfacetBtdf.f(wo, wi);
		return val;
	}

	__both__ BSDFSample sample(Vec3f wo, Sampler & sg) const {
		float u = sg.get1D();
		BSDFSample sample = {};
		if (u < pDiff) {	// reflection
			sample = diffuseBsdf.sample(wo, sg);
			sample.f *= (1 - specTrans) / pDiff;
			sample.pdf *= pDiff;
			if (pSpecRefl) sample.pdf += pSpecRefl * microfacetBrdf.pdf(wo, sample.wi);
		}
		else if (u < pDiff + pSpecRefl) {
			sample = microfacetBrdf.sample(wo, sg);
			sample.f *= (1 - specTrans) / pSpecRefl;
			sample.pdf *= pSpecRefl;
			if (pDiff) sample.pdf += pDiff * diffuseBsdf.pdf(wo, sample.wi);
		}
		else if (pSpecTrans > 0) {
			sample = microfacetBtdf.sample(wo, sg);
			sample.f *= specTrans / pSpecTrans;
			sample.pdf *= pSpecTrans;
		}
		return sample;
	}

	__both__ float pdf(Vec3f wo, Vec3f wi) const {
		float val = 0;
		if (pDiff) val += pDiff * diffuseBsdf.pdf(wo, wi);
		if (pSpecRefl) val += pSpecRefl * microfacetBrdf.pdf(wo, wi);
		if (pSpecTrans) val += pSpecTrans * microfacetBtdf.pdf(wo, wi);
		return val;
	}

	float pDiff{ 1 };
	float pSpecTrans{ 0 };
	float pSpecRefl{ 1 };

	float specTrans{ 0 };

	DiffuseBrdf diffuseBsdf;
	MicrofacetBrdf microfacetBrdf;
	MicrofacetBtdf microfacetBtdf;
};


KRR_NAMESPACE_END