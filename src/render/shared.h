#pragma once


#include "util/math_utils.h"
#include "raytracing.h"
#include "scene.h"

#define KRR_RT_RG(name) __raygen__##name
#define KRR_RT_MS(name) __miss__##name
#define KRR_RT_EX(name) __exception__##name
#define KRR_RT_CH(name) __closesthit__##name
#define KRR_RT_AH(name) __anyhit__##name
#define KRR_RT_IS(name) __intersection__##name
#define KRR_RT_DC(name) __direct_callable__##name
#define KRR_RT_CC(name) __continuation_callable__##name

#define EMPTY_DECLARE(name)                                                                        \
	extern "C" __global__ void name() { return; }

KRR_NAMESPACE_BEGIN


using namespace utils;

namespace shader {

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE = 1, RAY_TYPE_COUNT };

struct HitInfo {
	uint primitiveId;
	Vector3f barycentric;

	MeshData *mesh;
	Vector3f wo;
	uint hitKind;
};

struct ShadingData { // for use as per ray data, generated from ch
	Vector3f pos;
	Vector3f wo; // view direction
	Vector2f uv; // texture coords
	Frame frame;

	float IoR{ 1.5 };
	Color diffuse;			// diffuse reflectance
	Color specular;			// specular reflectance
	float specularTransmission{ 0 };
	float roughness{ 1 };	// linear roughness (alpha=roughness^2)
	float metallic{ 0 };	//
	float anisotropic{ 0 }; //

	Light light{ nullptr };
	MaterialType bsdfType;

	KRR_CALLABLE Interaction getInteraction() const { return Interaction(pos, wo, frame.N, uv); }

	/* This is a faster routine to get the bsdf type bypassing different bsdf implementations. */
	KRR_CALLABLE BSDFType getBsdfType() const {
		BSDFType type = BSDFType::BSDF_UNSET;
		switch (bsdfType) {
			case MaterialType::Diffuse:
				type = BSDFType::BSDF_DIFFUSE_REFLECTION;
				break;
			case MaterialType::Dielectric:
				type = roughness <= 1e-3f ? BSDF_SPECULAR : BSDF_GLOSSY;
				type = type | BSDF_REFLECTION | BSDF_TRANSMISSION;
				break;
			//case MaterialType::Principled:
			//	[[fallthrough]];
			case MaterialType::Disney:
				type = roughness <= 1e-3f ? BSDF_SPECULAR_REFLECTION : BSDF_GLOSSY_REFLECTION;
				if (diffuse.any() && specularTransmission < 1 && metallic < 1)
					type = type | BSDFType::BSDF_DIFFUSE_REFLECTION;
				if (specularTransmission > 0)
					type = type | BSDF_TRANSMISSION;
				break;
			default:
				printf("[ShadingData::getBsdfType] Unsupported BSDF.\n");
		}

		return type;
	}
};

// the following routines are used to encode 64-bit payload pointers
static KRR_DEVICE_FUNCTION void *unpackPointer(uint i0, uint i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void *ptr			= reinterpret_cast<void *>(uptr);
	return ptr;
}

static KRR_DEVICE_FUNCTION void packPointer(void *ptr, uint &i0, uint &i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0					= uptr >> 32;
	i1					= uptr & 0x00000000ffffffff;
}

template <typename T> static KRR_DEVICE_FUNCTION T *getPRD() {
	const uint u0 = optixGetPayload_0();
	const uint u1 = optixGetPayload_1();
	return reinterpret_cast<T *>(unpackPointer(u0, u1));
}
} // namespace shader

KRR_NAMESPACE_END