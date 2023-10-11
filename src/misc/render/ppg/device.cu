#include "render/shared.h"
#include "render/shading.h"
#include "render/wavefront/workqueue.h"

#include "ppg/ppg.h"
#include <optix_device.h>

using namespace krr;
KRR_NAMESPACE_BEGIN

extern "C" __constant__ LaunchParamsPPG launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, 3,							/* ray type and number of types */
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
	DCHECK_LT(rayIndex, launchParams.currentRayQueue->size());
	return (*launchParams.currentRayQueue)[rayIndex];
}

KRR_DEVICE_FUNCTION ShadowRayWorkItem getShadowRayWorkItem() {
	int rayIndex(optixGetLaunchIndex().x);
	DCHECK_LT(rayIndex, launchParams.shadowRayQueue->size());
	return (*launchParams.shadowRayQueue)[rayIndex];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo = getHitInfo();
	SurfaceInteraction& intr = *getPRD<SurfaceInteraction>();
	RayWorkItem r = getRayWorkItem();
	prepareSurfaceInteraction(intr, hitInfo);
	if (launchParams.mediumSampleQueue && r.ray.medium) {
		launchParams.mediumSampleQueue->push(r, intr, optixGetRayTmax());
		return;
	}
	if (intr.material == nullptr) {
		launchParams.nextRayQueue->push(intr.spawnRayTowards(r.ray.dir), r.ctx, r.thp, r.pu, r.pl,
										r.depth, r.pixelId, r.bsdfType);
		return;
	}
	if (intr.light) // push to hit ray queue if mesh has light
		launchParams.hitLightRayQueue->push(r, intr);
	if (r.thp.any()) // process material and push to material evaluation queue
		launchParams.scatterRayQueue->push(intr, r.thp, r.pu, r.depth, r.pixelId);
}

extern "C" __global__ void KRR_RT_AH(Closest)() { 
	if (alphaKilled())
		optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	launchParams.missRayQueue->push(getRayWorkItem());
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.traversable, r.ray, M_FLOAT_INF,
		0, OPTIX_RAY_FLAG_NONE, (void*)&intr);
}

extern "C" __global__ void KRR_RT_AH(Shadow)() { 
	if (alphaKilled())
		optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Shadow)() { optixSetPayload_0(1); }

extern "C" __global__ void KRR_RT_RG(Shadow)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t visible{0};
	traceRay(launchParams.traversable, r.ray, r.tMax, 1,
			 OptixRayFlags( OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		visible);
	if (visible) {
		Color contrib = r.Ld / (r.pl + r.pu).mean();
		launchParams.pixelState->addRadiance(r.pixelId, contrib);
		if (launchParams.enableTraining)
			launchParams.guidedState->recordRadiance(r.pixelId, contrib);
	}
}

extern "C" __global__ void KRR_RT_CH(ShadowTr)() {
	HitInfo hitInfo			 = getHitInfo();
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	prepareSurfaceInteraction(intr, hitInfo);
}

extern "C" __global__ void KRR_RT_AH(ShadowTr)() {
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(ShadowTr)() { optixSetPayload_2(1); }

extern "C" __global__ void KRR_RT_RG(ShadowTr)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r		= getShadowRayWorkItem();
	SurfaceInteraction intr = {};
	uint u0, u1;
	packPointer(&intr, u0, u1);
	traceTransmittance(r, intr, launchParams.pixelState, [&](Ray ray, float tMax) -> bool {
		uint32_t visible = 0;
		traceRay(launchParams.traversable, ray, tMax, 2, OPTIX_RAY_FLAG_NONE, u0, u1, visible);
		return visible;
	});
}


KRR_NAMESPACE_END