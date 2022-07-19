#pragma once

#include "common.h"

#define KRR_USE_DISNEY			1
#define KRR_USE_FRESNEL_BLEND	1
#define KRR_USE_SCHLICK_FRESNEL 1

KRR_NAMESPACE_BEGIN

namespace bsdf{

	enum BxDFType {
		BSDF_REFLECTION = 1 << 0,
		BSDF_TRANSMISSION = 1 << 1,
		BSDF_DIFFUSE = 1 << 2,
		BSDF_GLOSSY = 1 << 3,
		BSDF_SPECULAR = 1 << 4,
		BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION,
	};

	struct BSDFSample {
		Color f{};
		vec3f wi;
		float pdf = 0;
		uint flags;
		bool valid = true;
	};
}

KRR_NAMESPACE_END