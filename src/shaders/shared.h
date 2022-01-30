#pragma once
//#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_device.h>
#include <optix_types.h>

#include "math/math.h"
#include "scene.h"

#define KRR_RT_RG(name) __raygen__ ## name
#define KRR_RT_MS(name) __miss__ ## name
#define KRR_RT_EX(name) __exception__ ## name
#define KRR_RT_CH(name) __closesthit__ ## name
#define KRR_RT_AH(name) __anyhit__ ## name
#define KRR_RT_IS(name) __intersection__ ## name
#define KRR_RT_DC(name) __direct_callable__ ## name
#define KRR_RT_CC(name) __continuation_callable__ ## name

namespace krr{
	namespace shader {

		// the following routines are used to encode 64-bit payload pointers
		static KRR_DEVICE_FUNCTION
			void* unpackPointer(uint32_t i0, uint32_t i1)
		{
			const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
			void* ptr = reinterpret_cast<void*>(uptr);
			return ptr;
		}

		static KRR_DEVICE_FUNCTION
			void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
		{
			const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
			i0 = uptr >> 32;
			i1 = uptr & 0x00000000ffffffff;
		}

		template<typename T>
		static KRR_DEVICE_FUNCTION T* getPRD()
		{
			const uint32_t u0 = optixGetPayload_0();
			const uint32_t u1 = optixGetPayload_1();
			return reinterpret_cast<T*>(unpackPointer(u0, u1));
		}
	}

}