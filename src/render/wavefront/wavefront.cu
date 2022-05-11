#include "render/shared.h"
#include "render/shading.cuh"
#include "render/wavefront/wavefront.h"
#include "render/wavefront/workqueue.h"

#include <optix_device.h>

using namespace krr;
KRR_NAMESPACE_BEGIN

extern "C" __constant__ LaunchParams launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, 1,			/* ray type and number of types */
		rayType,							/* miss SBT index */
		std::forward<Args>(payload)...);	/* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, void* payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, u0, u1);
}

KRR_DEVICE_FUNCTION RayWorkItem getRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK(rayIndex < launchParams.currentRayQueue->size());
	return (*launchParams.currentRayQueue)[rayIndex];
}

KRR_DEVICE_FUNCTION ShadowRayWorkItem getShadowRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK(rayIndex < launchParams.shadowRayQueue->size());
	return (*launchParams.shadowRayQueue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo = getHitInfo();
	ShadingData& sd = *getPRD<ShadingData>();
	RayWorkItem r = getRayWorkItem();
	sd.miss = false;
	Material& material = (*launchParams.sceneData.materials)[hitInfo.mesh->materialId];
	prepareShadingData(sd, hitInfo, material);
	if (sd.light) {		// push to hit ray queue if mesh has light
		HitLightWorkItem w = {};
		w.light = sd.light;
		w.p = sd.pos;
		w.n = sd.N;
		w.wo = sd.wo;
		w.uv = sd.uv;
		w.pixelId = r.pixelId;
		w.thp = r.thp;
		launchParams.hitLightRayQueue->push(w);
	}
	// process material and push to material evaluation queue (if eligible)
	if (any(r.thp)) {
		ScatterRayWorkItem w = {};
		w.pixelId = r.pixelId;
		w.thp = r.thp;
		w.sd = sd;
		w.depth = r.depth;
		launchParams.scatterRayQueue->push(w);
	}
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	ShadingData* sd = getPRD<ShadingData>();
	sd->miss = true;
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	int rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = getRayWorkItem();
	ShadingData sd = {};
	traceRay(launchParams.traversable, r.ray, KRR_RAY_TMAX,
		0, OPTIX_RAY_FLAG_DISABLE_ANYHIT, (void*)&sd);
	if (sd.miss) {	// push to miss ray queue
		launchParams.missRayQueue->push(r);
	}
}

extern "C" __global__ void KRR_RT_AH(Shadow)() {
	ShadingData* sd = getPRD<ShadingData>();
	sd->miss = false;
}

extern "C" __global__ void KRR_RT_MS(Shadow)() {
	return;
}

extern "C" __global__ void KRR_RT_RG(Shadow)() {
	int rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
	ShadingData sd = {};
	traceRay(launchParams.traversable, r.ray, KRR_RAY_TMAX,
		0, OPTIX_RAY_FLAG_DISABLE_ANYHIT, (void*)&sd);
}

KRR_NAMESPACE_END