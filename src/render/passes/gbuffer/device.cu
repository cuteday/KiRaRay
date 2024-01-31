#include "device.h"
#include "render/shared.h"
#include "render/shading.h"

#include <optix_device.h>

NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters <GBufferPass> launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(traversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						 /* all visible */
			   flags, rayType, 2,						/* ray type and number of types */
			   rayType,											 /* miss SBT index */
			   std::forward<Args>(payload)...);			/* (unpacked pointers to) payloads */
}

KRR_RT_KERNEL KRR_RT_CH(Primary)() {

}

KRR_RT_KERNEL KRR_RT_MS(Primary)() {

}

KRR_RT_KERNEL KRR_RT_RG(Primary)() {
	const Vector3ui launchIndex = optixGetLaunchIndex();
	const Vector2ui pixel		= {launchIndex[0], launchIndex[1]};
	const uint32_t pixelIndex	= pixel[0] + pixel[1] * launchParams.frameSize[0];

	PCGSampler pcgSampler;
	pcgSampler.setPixelSample(pixel, (uint32_t) launchParams.frameIndex);
	pcgSampler.advance(pixelIndex * 256);

	Sampler sampler = &pcgSampler;
	Ray ray = launchParams.cameraData.getRay(pixel, launchParams.frameSize, sampler);

	traceRay(launchParams.traversable, ray, M_FLOAT_INF, 0, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
}

NAMESPACE_END(krr)