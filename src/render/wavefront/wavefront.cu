#include "render/shared.h"
#include "render/shading.cuh"
#include "render/wavefront/wavefront.h"
#include "render/wavefront/workqueue.h"
#include "render/raytracing.cuh"

#include <optix_device.h>

using namespace krr;
KRR_NAMESPACE_BEGIN

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo = getHitInfo();
	ShadingData& sd = *getPRD<ShadingData>();
	sd.miss = false;
	Material& material = launchParams.sceneData.materials[hitInfo.mesh->materialId];
	prepareShadingData(sd, hitInfo, material);
	
	if (sd.light) {	// push to hit ray queue if mesh has light

	}
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	ShadingData* sd = getPRD<ShadingData>();
	sd->miss = true;
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	int rayIndex(optixGetLaunchIndex().x);
	//printf("current thread id: %d, total rays: %d\n", rayIndex, launchParams.currentRayQueue->size());
	if (rayIndex >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = (*launchParams.currentRayQueue)[rayIndex];
	Ray ray = r.ray;

	ShadingData sd = {};
	traceRay(launchParams.traversable, ray, KRR_RAY_TMAX,
		0, OPTIX_RAY_FLAG_DISABLE_ANYHIT, (void*)&sd);
	if (sd.miss) {	// push to miss ray queue
		launchParams.missRayQueue->push(r);
		//printf("pushing to missing ray queue, current queue size: %d\n", launchParams.missRayQueue->size());
	}
}

extern "C" __global__ void KRR_RT_AH(Shadow)() {

}

extern "C" __global__ void KRR_RT_MS(Shadow)() {

}

extern "C" __global__ void KRR_RT_RG(Shadow)() {
	int rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.shadowRayQueue->size()) return;
	RayWorkItem r = (*launchParams.currentRayQueue)[rayIndex];
	Ray ray = r.ray;

}

KRR_NAMESPACE_END