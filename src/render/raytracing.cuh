#pragma once
#include "common.h"

#include <optix_device.h>

KRR_NAMESPACE_BEGIN

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, RAY_TYPE_COUNT,			/* ray type and number of types */
		rayType,							/* miss SBT index */
		std::forward<Args>(payload)...);	/* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, void* payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, u0, u1);
}

template <typename... Args>
KRR_DEVICE_FUNCTION bool traceShadowRay(OptixTraversableHandle traversable,
	Ray ray, float tMax) {
	ShadowRayData sd = { false };
	OptixRayFlags flags = (OptixRayFlags)(OPTIX_RAY_FLAG_DISABLE_ANYHIT);
	uint u0, u1;
	packPointer(&sd, u0, u1);
	traceRay(traversable, ray, tMax, (int)SHADOW_RAY_TYPE, flags, u0, u1);
	return sd.visible;
}

KRR_NAMESPACE_END