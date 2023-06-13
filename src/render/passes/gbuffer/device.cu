#include "device.h"
#include "render/shading.h"

using namespace krr;
KRR_NAMESPACE_BEGIN

extern "C" __constant__ LaunchParamsGBuffer launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(traversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						 /* all visible */
			   flags, rayType, RAY_TYPE_COUNT,					 /* ray type and number of types */
			   rayType,											 /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

extern "C" __global__ void KRR_RT_CH(Primary)() {

}

extern "C" __global__ void KRR_RT_MS(Primary)() {

}

extern "C" __global__ void KRR_RT_RG(Primary)() {
	Vector3ui launchIndex = optixGetLaunchIndex();
	Vector2ui pixel		  = {launchIndex[0], launchIndex[1]};
}

KRR_NAMESPACE_END