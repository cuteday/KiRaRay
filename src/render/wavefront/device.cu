#include "render/shared.h"
#include "render/shading.h"
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

KRR_DEVICE_FUNCTION int getRayId() { return optixGetLaunchIndex().x; }

KRR_DEVICE_FUNCTION RayWorkItem getRayWorkItem() {
	DCHECK_LT(getRayId(), launchParams.currentRayQueue->size());
	return (*launchParams.currentRayQueue)[getRayId()];
}

KRR_DEVICE_FUNCTION ShadowRayWorkItem getShadowRayWorkItem() {
	DCHECK_LT(getRayId(), launchParams.shadowRayQueue->size());
	return (*launchParams.shadowRayQueue)[getRayId()];
}

extern "C" __global__ void KRR_RT_CH(Closest)() {
	HitInfo hitInfo			  = getHitInfo();
	SurfaceInteraction &intr  = *getPRD<SurfaceInteraction>();
	RayWorkItem r			  = getRayWorkItem();
	int pixelId				  = launchParams.currentRayQueue->pixelId[getRayId()];
	SampledWavelengths lambda = launchParams.pixelState->lambda[pixelId];
	intr.medium				  = r.ray.medium;	// if this surface is inside a medium
	prepareSurfaceInteraction(intr, hitInfo, lambda);
	if (launchParams.mediumSampleQueue && r.ray.medium) {
		launchParams.mediumSampleQueue->push(r, intr, optixGetRayTmax());
		return;
	}
	if (intr.material == nullptr) {
		launchParams.nextRayQueue->push(intr.spawnRayTowards(r.ray.dir), r.ctx, r.thp, r.pu, r.pl,
										r.depth, r.pixelId, r.bsdfType);
		return;
	}
	if (intr.light) 	// push to hit ray queue if mesh has light
		launchParams.hitLightRayQueue->push(r, intr);
	if (r.thp.any()) 	// process material and push to material evaluation queue
		launchParams.scatterRayQueue->push(intr, r.thp, r.pu, r.depth, r.pixelId);
}

extern "C" __global__ void KRR_RT_AH(Closest)() { 
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	const RayWorkItem &r = getRayWorkItem();
	if (launchParams.mediumSampleQueue && r.ray.medium)
		launchParams.mediumSampleQueue->push(r, M_FLOAT_INF); 
	else launchParams.missRayQueue->push(r);
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	if (getRayId() >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.traversable, r.ray, M_FLOAT_INF,
		0, OPTIX_RAY_FLAG_NONE, (void*)&intr);
}

extern "C" __global__ void KRR_RT_AH(Shadow)() { 
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Shadow)() { optixSetPayload_0(1); }

extern "C" __global__ void KRR_RT_RG(Shadow)() {
	if (getRayId() >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t visible{0};
	traceRay(launchParams.traversable, r.ray, r.tMax, 1,
			 OptixRayFlags( OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		visible);
	if (visible) launchParams.pixelState->addRadiance(r.pixelId, r.Ld / (r.pl + r.pu).mean());
}

extern "C" __global__ void KRR_RT_CH(ShadowTr)() {
	HitInfo hitInfo			 = getHitInfo();
	int pixelId				  = launchParams.shadowRayQueue->pixelId[getRayId()]; 
	SampledWavelengths lambda = launchParams.pixelState->lambda[pixelId];
	SurfaceInteraction &intr = *getPRD<SurfaceInteraction>();
	prepareSurfaceInteraction(intr, hitInfo, lambda);
}

extern "C" __global__ void KRR_RT_AH(ShadowTr)() {
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(ShadowTr)() { optixSetPayload_2(1); }

extern "C" __global__ void KRR_RT_RG(ShadowTr)() {
	if (getRayId() >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
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