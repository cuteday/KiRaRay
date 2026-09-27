#include "render/shared.h"
#include "render/shading.h"
#include "render/wavefront/wavefront.h"
#include "render/wavefront/workqueue.h"

#include <optix_device.h>

NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters <WavefrontPathTracer> launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, OptixVisibilityMask mask, Args &&... payload) {
	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, ray.time,				/* ray time val min max */
		mask,
		flags,
		rayType, 3,							/* ray type and number of types */
		rayType,							/* miss SBT index */
		std::forward<Args>(payload)...);	/* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, OptixVisibilityMask mask, void* payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, mask, u0, u1);
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

KRR_RT_KERNEL KRR_RT_CH(Closest)() {
	HitInfo hitInfo			  = getHitInfo();
	SurfaceInteraction &intr  = *getPRD<SurfaceInteraction>();
	RayWorkItem r			  = getRayWorkItem();
	int pixelId				  = launchParams.currentRayQueue->pixelId[getRayId()];
	SampledWavelengths &lambda = launchParams.pixelState->lambda[pixelId];
	prepareSurfaceInteraction(intr, hitInfo, r.ray, lambda, launchParams.authoredMaterials);
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

KRR_RT_KERNEL KRR_RT_AH(Closest)() { 
	if (alphaKilled(getHitInfo(), launchParams.authoredMaterials)) optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_MS(Closest)() {
	const RayWorkItem &r = getRayWorkItem();
	if (launchParams.mediumSampleQueue && r.ray.medium)
		launchParams.mediumSampleQueue->push(r, M_FLOAT_INF); 
	else launchParams.missRayQueue->push(r);
}

KRR_RT_KERNEL KRR_RT_RG(Closest)() {
	if (getRayId() >= launchParams.currentRayQueue->size()) return;
	RayWorkItem r = getRayWorkItem();
	SurfaceInteraction intr = {};
	traceRay(launchParams.traversable, r.ray, M_FLOAT_INF, 0, OPTIX_RAY_FLAG_NONE,
		r.depth == 0 ? 1 : 255, (void *) &intr);
}

KRR_RT_KERNEL KRR_RT_AH(Shadow)() { 
	/* We are here since we did not enable medium rendering, so safely ignore null-material. */
	const HitInfo &hitInfo = getHitInfo();
	if (hitInfo.instance->mesh->material == nullptr || alphaKilled(hitInfo, launchParams.authoredMaterials))
		optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_MS(Shadow)() { optixSetPayload_0(1); }

KRR_RT_KERNEL KRR_RT_RG(Shadow)() {
	if (getRayId() >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
	uint32_t visible{0};
	traceRay(launchParams.traversable, r.ray, r.tMax, 1,
			 OptixRayFlags( OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT),
		255, visible);
	if (visible) launchParams.pixelState->addRadiance(r.pixelId, r.Ld / (r.pl + r.pu).mean());
}

KRR_RT_KERNEL KRR_RT_CH(ShadowTr)() {
	HitInfo hitInfo			  = getHitInfo();
	ShadowRayWorkItem r		  = getShadowRayWorkItem();
	int pixelId				  = launchParams.shadowRayQueue->pixelId[getRayId()]; 
	SampledWavelengths &lambda = launchParams.pixelState->lambda[pixelId];
	SurfaceInteraction &intr  = *getPRD<SurfaceInteraction>();
	prepareSurfaceInteraction(intr, hitInfo, r.ray, lambda, launchParams.authoredMaterials);
}

KRR_RT_KERNEL KRR_RT_AH(ShadowTr)() {
	if (alphaKilled(getHitInfo(), launchParams.authoredMaterials)) optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_MS(ShadowTr)() { optixSetPayload_2(1); }

KRR_RT_KERNEL KRR_RT_RG(ShadowTr)() {
	if (getRayId() >= launchParams.shadowRayQueue->size()) return;
	ShadowRayWorkItem r = getShadowRayWorkItem();
	SurfaceInteraction intr = {};
	uint u0, u1;
	packPointer(&intr, u0, u1);
	traceTransmittance(r, intr, launchParams.pixelState, [&](Ray ray, float tMax) -> bool {
		uint32_t visible = 0;
		traceRay(launchParams.traversable, ray, tMax, 2, OPTIX_RAY_FLAG_NONE, 255, u0, u1, visible);
		return visible;
	});
}

NAMESPACE_END(krr)
