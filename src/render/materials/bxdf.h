#pragma once

#include "common.h"
#include "render/spectrum.h"

#define KRR_USE_DISNEY			1

NAMESPACE_BEGIN(krr)

enum BSDFType {
	BSDF_UNSET					= 0,
	BSDF_NULL					= 1 << 0,
	BSDF_REFLECTION				= 1 << 1,
	BSDF_TRANSMISSION			= 1 << 2,
	BSDF_DIFFUSE				= 1 << 3,
	BSDF_GLOSSY					= 1 << 4,
	BSDF_SPECULAR				= 1 << 5,
	BSDF_DIFFUSE_REFLECTION		= BSDF_DIFFUSE | BSDF_REFLECTION,
	BSDF_SPECULAR_REFLECTION	= BSDF_SPECULAR | BSDF_REFLECTION,
	BSDF_GLOSSY_REFLECTION		= BSDF_GLOSSY | BSDF_REFLECTION,
	BSDF_DIFFUSE_TRANSMISSION	= BSDF_DIFFUSE | BSDF_TRANSMISSION,
	BSDF_SPECULAR_TRANSMISSION	= BSDF_SPECULAR | BSDF_TRANSMISSION,
	BSDF_GLOSSY_TRANSMISSION	= BSDF_GLOSSY | BSDF_TRANSMISSION,
	/* Type combinations */
	BSDF_SMOOTH				= BSDF_DIFFUSE | BSDF_GLOSSY,
	BSDF_DELTA				= BSDF_SPECULAR | BSDF_NULL,
	BSDF_MATERIAL_TYPES		= BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR,
	BSDF_SCATTER_TYPES		= BSDF_REFLECTION | BSDF_TRANSMISSION,
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION,
};

enum class TransportMode {
	Radiance,
	Importance
};

KRR_ENUM_OPERATORS(BSDFType)

struct BSDFSample {
	Spectrum f{};
	Vector3f wi;
	float pdf = 0;
	BSDFType flags;

	KRR_CALLABLE BSDFSample() = default;
	KRR_CALLABLE BSDFSample(Spectrum f, Vector3f wi, float pdf, BSDFType flags)
		: f(f), wi(wi), pdf(pdf), flags(flags) {}

	KRR_CALLABLE bool isDelta() const { return flags & BSDF_SPECULAR; }
	KRR_CALLABLE bool isDiffuse() const { return flags & BSDF_DIFFUSE; }
	KRR_CALLABLE bool isGlossy() const { return flags & BSDF_GLOSSY; }
	KRR_CALLABLE bool isReflective() const { return flags & BSDF_REFLECTION; }
	KRR_CALLABLE bool isTransmissive() const { return flags & BSDF_TRANSMISSION; }
	KRR_CALLABLE bool isNonSpecular() const { return flags & BSDF_SMOOTH; }
						
};

NAMESPACE_END(krr)