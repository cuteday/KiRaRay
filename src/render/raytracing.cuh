#pragma once
#include "common.h"

#include <optix_device.h>
#include "math/math.h"

KRR_NAMESPACE_BEGIN

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, to_float3(ray.origin), to_float3(ray.dir),
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, RAY_TYPE_COUNT,			/* ray type and number of types */
		rayType,							/* miss SBT index */
		std::forward<Args>(payload)...);	/* (unpacked pointers to) payloads */
}


KRR_NAMESPACE_END