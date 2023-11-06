#include "bdpt.h"
#include "render/shared.h"
#include "render/shading.h"
#include "util/hash.h"
#include "util/math_utils.h"

#include <optix_device.h>

using namespace krr;	
KRR_NAMESPACE_BEGIN

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(traversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						 /* all visible */
			   flags, rayType, 1,								 /* ray type and number of types */
			   rayType,											 /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION float G(const Vertex& v1, const Vertex& v2) {
	return 0;
}

KRR_DEVICE_FUNCTION int ExtendPath(Sampler sampler, int maxDepth,
	Vertex* path, TransportMode mode) {
	return 0;
}

KRR_DEVICE_FUNCTION RGB Connect(Vertex* lightVertices, Vertex* cameraVertices,
	uint s, uint t, LightSampler lightSampler, Sampler sampler) {
	return 0;
}

extern "C" __global__ void KRR_RT_RG(GenerateCameraSubpath)() {
	uint workId(optixGetLaunchIndex().x);
}

extern "C" __global__ void KRR_RT_RG(GenerateLightSubpath)() {
	uint workId(optixGetLaunchIndex().x);
}

extern "C" __global__ void KRR_RT_RG(Connect)() {
	uint workId(optixGetLaunchIndex().x);
}

KRR_NAMESPACE_END