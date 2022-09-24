#pragma once

#include "common.h"
#include "math/math.h"

#define KRR_USE_DISNEY			1
#define KRR_USE_FRESNEL_BLEND	1
#define KRR_USE_SCHLICK_FRESNEL 1

KRR_NAMESPACE_BEGIN

namespace bsdf{

	enum BSDFType {
		BSDF_REFLECTION = 1 << 0,
		BSDF_TRANSMISSION = 1 << 1,
		BSDF_DIFFUSE = 1 << 2,
		BSDF_GLOSSY = 1 << 3,
		BSDF_SPECULAR = 1 << 4,
		BSDF_DIFFUSE_REFLECTION = BSDF_DIFFUSE | BSDF_REFLECTION,
		BSDF_SPECULAR_REFLECTION = BSDF_SPECULAR | BSDF_REFLECTION,
		BSDF_GLOSSY_REFLECTION = BSDF_GLOSSY | BSDF_REFLECTION,
		BSDF_DIFFUSE_TRANSMISSION = BSDF_DIFFUSE | BSDF_TRANSMISSION,
		BSDF_SPECULAR_TRANSMISSION = BSDF_SPECULAR | BSDF_TRANSMISSION,
		BSDF_GLOSSY_TRANSMISSION = BSDF_GLOSSY | BSDF_TRANSMISSION,
		BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION,
	};

	KRR_CALLABLE BSDFType operator|(BSDFType a, BSDFType b) { return BSDFType((int) a | (int) b); }
	KRR_CALLABLE BSDFType operator&(BSDFType a, BSDFType b) { return BSDFType((int) a & (int) b); }

	struct BSDFSample {
		Color f{};
		Vector3f wi;
		float pdf = 0;
		BSDFType flags;
		bool valid = true;

		KRR_CALLABLE BSDFSample() = default;
		KRR_CALLABLE BSDFSample(Color f, Vector3f wi, float pdf, BSDFType flags)
			: f(f), wi(wi), pdf(pdf), flags(flags) {}
	};
}

KRR_NAMESPACE_END