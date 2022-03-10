#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "scene.h"

#define KRR_RT_RG(name) __raygen__ ## name
#define KRR_RT_MS(name) __miss__ ## name
#define KRR_RT_EX(name) __exception__ ## name
#define KRR_RT_CH(name) __closesthit__ ## name
#define KRR_RT_AH(name) __anyhit__ ## name
#define KRR_RT_IS(name) __intersection__ ## name
#define KRR_RT_DC(name) __direct_callable__ ## name
#define KRR_RT_CC(name) __continuation_callable__ ## name

#define EMPTY_DECLARE(name) extern "C" __global__ void name () { return; }

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace math::utils;

namespace shader {
		
	enum {
		RADIANCE_RAY_TYPE = 0,
		SHADOW_RAY_TYPE = 1,
		RAY_TYPE_COUNT
	};

	struct ShadingFrame {
		vec3f N;
		vec3f T;
		vec3f B;
	};

	struct HitInfo {
		uint primitiveId;
		vec3f barycentric;
			
		//Mesh* mesh;
		MeshData* mesh;
		vec3f wo;
		uint hitKind;
	};

	struct ShadingData {		// for use as per ray data, generated from ch
		using ShadingModel = Material::ShadingModel;
		using BsdfType = Material::BsdfType;
			
		vec3f pos;
		vec3f wo;				// view direction
		vec2f uv;				// texture coords
		vec3f geoN;				// geometry normal [on the front-facing side]

		vec3f N;				// shading normal [flipped if back-facing]
		vec3f T;				// tangent
		vec3f B;				// bitangent

		float IoR{ 1.5 };
		vec3f diffuse;			// diffuse reflectance
		vec3f specular;			// specular reflectance

		vec3f transmission{ 1 };// transmission color (shared by diffuse and specular for now)
		float diffuseTransmission{ 0 };
		float specularTransmission{ 0 };

		float roughness{ 1 };		// linear roughness (alpha=roughness^2)
		float metallic{ 0 };		// 
		float anisotropic{ 0 };		// 

		Light light{ nullptr };
		bool miss = false;		// not valid if missing, or ?
		bool frontFacing;		// shading normal and geo normal same dir?
			
		//ShadingModel shadingModel = ShadingModel::MetallicRoughness;
		BsdfType bsdfType = BsdfType::Diffuse;
		uint flags = 0;			// user custom flags?

		__both__ vec3f fromLocal(vec3f v) const {
			return T * v.x + B * v.y + N * v.z;
		}

		__both__ vec3f  toLocal(vec3f v) const {
			return { dot(T, v), dot(B, v), dot(N, v) };
		}
	};

	// the following routines are used to encode 64-bit payload pointers
	static KRR_DEVICE_FUNCTION
		void* unpackPointer(uint i0, uint i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static KRR_DEVICE_FUNCTION
		void  packPointer(void* ptr, uint& i0, uint& i1)
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static KRR_DEVICE_FUNCTION T* getPRD()
	{
		const uint u0 = optixGetPayload_0();
		const uint u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}
}

KRR_NAMESPACE_END