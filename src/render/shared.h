#pragma once

#include "math/math.h"
#include "math/utils.h"
#include "raytracing.h"
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

	struct HitInfo {
		uint primitiveId;
		Vec3f barycentric;

		MeshData* mesh;
		Vec3f wo;
		uint hitKind;
	};

	struct ShadingData {		// for use as per ray data, generated from ch
		Vec3f pos;
		Vec3f wo;				// view direction
		Vec2f uv;				// texture coords
		Vec3f geoN;				// geometry normal [on the front-facing side]

		Frame frame;

		float IoR{ 1.5 };
		Color diffuse;			// diffuse reflectance
		Color specular; // specular reflectance

		Color transmission{ Color::Ones() }; // transmission color (shared by diffuse and specular for now)
		float diffuseTransmission{ 0 };
		float specularTransmission{ 0 };

		float roughness{ 1 };		// linear roughness (alpha=roughness^2)
		float metallic{ 0 };		// 
		float anisotropic{ 0 };		// 

		Light light{ nullptr };
		bool miss = false;			// not valid if missing, or ?
			
		BsdfType bsdfType;
		uint flags;					// user custom flags?

		__both__ Interaction getInteraction() const {
			return Interaction(pos, wo, frame.N, uv);
		}
	};

	// the following routines are used to encode 64-bit payload pointers
	static KRR_DEVICE_FUNCTION void* unpackPointer(uint i0, uint i1) {
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static KRR_DEVICE_FUNCTION void packPointer(void* ptr, uint& i0, uint& i1) {
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static KRR_DEVICE_FUNCTION T* getPRD() {
		const uint u0 = optixGetPayload_0();
		const uint u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}
}

KRR_NAMESPACE_END