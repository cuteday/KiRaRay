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
		rayType, 2,							/* ray type and number of types */
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
		MediumSampleWorkItem w = {};
		w.depth				   = r.depth;
		w.ray				   = r.ray;
		w.thp				   = r.thp;
		w.pixelId			   = r.pixelId;
		w.intr				   = intr;
		w.tMax				   = optixGetRayTmax();
		launchParams.mediumSampleQueue->push(w);
		return;
	}
	if (intr.light) {		// push to hit ray queue if mesh has light
		HitLightWorkItem w = {};
		w.light			   = intr.light;
		w.ctx			   = r.ctx;
		w.p				   = intr.p;
		w.n				   = intr.n;
		w.wo			   = intr.wo;
		w.uv			   = intr.uv;
		w.depth			   = r.depth;
		w.pixelId		   = r.pixelId;
		w.thp			   = r.thp;
		w.pdf			   = r.pdf;
		w.bsdfType		   = r.bsdfType;
		launchParams.hitLightRayQueue->push(w);
	}
	// process material and push to material evaluation queue (if eligible)
	if (any(r.thp)) {
		ScatterRayWorkItem w = {};
		w.pixelId			 = r.pixelId;
		w.thp				 = r.thp;
		w.intr				 = intr;
		w.depth				 = r.depth;
		launchParams.scatterRayQueue->push(w);
	}
}

extern "C" __global__ void KRR_RT_AH(Closest)() { 
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Closest)() {
	const RayWorkItem &r = getRayWorkItem();
	if (launchParams.mediumSampleQueue && r.ray.medium)
		launchParams.mediumSampleQueue->push(r); 
	else launchParams.missRayQueue->push(r);
}

extern "C" __global__ void KRR_RT_RG(Closest)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.traversable, r.ray, KRR_RAY_TMAX,
		0, OPTIX_RAY_FLAG_NONE, (void*)&intr);
}

extern "C" __global__ void KRR_RT_AH(Shadow)() { 
	if (alphaKilled()) optixIgnoreIntersection();
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
	if (visible) launchParams.pixelState->addRadiance(r.pixelId, r.Li * r.a);
}

extern "C" __global__ void KRR_RT_AH(ShadowTr)() {
	if (alphaKilled()) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(ShadowTr)() { optixSetPayload_2(1); }

extern "C" __global__ void KRR_RT_RG(ShadowTr)() {
	uint rayIndex(optixGetLaunchIndex().x);
	if (rayIndex >= launchParams.shadowRayQueue->size()) return;
}

KRR_NAMESPACE_END