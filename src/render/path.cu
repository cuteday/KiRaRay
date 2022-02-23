#include <optix_device.h>
#include <optix.h>

#include "math/utils.h"
#include "path.h"
#include "shared.h"
#include "shading.h"

using namespace krr;	// this is needed or nvcc can't recognize the launchParams external var.

KRR_NAMESPACE_BEGIN

namespace {
	constexpr float kMaxDistance = 1e9f;
}

using namespace math;
using namespace math::utils;
using namespace shader;
using namespace types;

extern "C" __constant__ LaunchParamsPT launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, RAY_TYPE_COUNT,	/* ray type and number of types */
		rayType,					/* miss SBT index */
		std::forward<Args>(payload)...);
}

template <typename... Args>
KRR_DEVICE_FUNCTION bool traceShadowRay(OptixTraversableHandle traversable, 
	Ray ray, float distance) {
	ShadowRayData shadow = { false };
	OptixRayFlags flags = (OptixRayFlags)(OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
	uint u0, u1;
	packPointer(&shadow, u0, u1);
	traceRay(traversable, ray, distance, (int)SHADOW_RAY_TYPE,
		flags, u0, u1);
	return shadow.visible;
}

KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
	vec3f Li = sd.emission;
	float weight = 1.f;
	path.L += Li * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void handleMiss(const ShadingData& sd, PathData& path) {
	vec3f wi = normalize(path.dir);
	vec3f Li = launchParams.envLight.eval(wi);
	float weight = 1.f;
	if (launchParams.NEE) {
		float bsdfWeight = path.pdf;
		float lightWeight = launchParams.envLight.pdf(wi);
		float weight = evalMIS(bsdfWeight, lightWeight);
	}
	path.L += weight * path.throughput * Li;
}

KRR_DEVICE_FUNCTION void evalDirect(const ShadingData& sd, PathData& path) {
	// currently evaluate environment map only.
	LightSample ls = {};
	vec2f u = path.sampler.get2D();
	vec3f woLocal = sd.toLocal(sd.wo);

	launchParams.envLight.sample(u, ls);
	vec3f wiWorld = sd.fromLocal(ls.wi);
	if (dot(wiWorld, sd.N) < 0) return;

	float lightPdf = ls.pdf;
	float bsdfPdf = BxDF::pdf(sd, woLocal, ls.wi, (int)sd.bsdfType);
	vec3f bsdfVal = BxDF::f(sd, woLocal, ls.wi, (int)sd.bsdfType);
	float weight = evalMIS(lightPdf, bsdfPdf);

	bool visible = traceShadowRay(launchParams.traversable, { path.pos, wiWorld }, kMaxDistance);

	if(visible)
		path.L += path.throughput * bsdfVal * weight * ls.Li;
}

KRR_DEVICE_FUNCTION bool generateScatter(const ShadingData& sd, PathData& path) {
	BSDFSample sample = {};

	// how to eliminate branches here to improve performance?
	vec3f wo = sd.toLocal(sd.wo);
	sample = BxDF::sample(sd, wo, path.sampler, (uint)sd.bsdfType);
	if (sample.pdf == 0) return false;
	if (sample.f == vec3f(0)) return false;
		
	vec3f wi = sd.fromLocal(sample.wi);
	path.pos = sd.pos + sd.N * 1e-3f;
	path.dir = wi;
	path.pdf = max(sample.pdf, 1e-6f);
	path.throughput *= sample.f * fabs(dot(wi, sd.N)) / path.pdf;
	return true;
}

extern "C" __global__ void KRR_RT_CH(Radiance)(){
	HitInfo hitInfo = {};
	vec2f barycentric = optixGetTriangleBarycentrics();
	hitInfo.primitiveId = optixGetPrimitiveIndex();
	hitInfo.mesh = (MeshData*)optixGetSbtDataPointer();
	hitInfo.wo = -normalize(vec3f(optixGetWorldRayDirection()));
	hitInfo.hitKind = optixGetHitKind();
	hitInfo.barycentric = { 1 - barycentric.x - barycentric.y, barycentric.x, barycentric.y };

	ShadingData& sd = *getPRD<ShadingData>();
	sd.miss = false;
	Material& material = launchParams.sceneData.materials[hitInfo.mesh->materialId];

	prepareShadingData(sd, hitInfo, material);
	//sd.emission = 0.2 + 0.8 * dot(sd.N, sd.wo);
}

extern "C" __global__ void KRR_RT_AH(Radiance)(){
	// currently skipped
	return;
}

extern "C" __global__ void KRR_RT_MS(Radiance)(){
	ShadingData &sd = *getPRD<ShadingData>();
	sd.miss = true;
}

extern "C" __global__ void KRR_RT_CH(ShadowRay)() {
	return;
}

extern "C" __global__ void KRR_RT_AH(ShadowRay)() {
	return;
}

extern "C" __global__ void KRR_RT_MS(ShadowRay)() {
	ShadowRayData& shadow = *getPRD<ShadowRayData>();
	shadow.visible = true;
}

KRR_DEVICE_FUNCTION void tracePath(PathData& path) {
	for (uint depth = 0; depth < launchParams.maxDepth; depth++) {
		ShadingData sd = {};
		uint u0, u1;
		packPointer(&sd, u0, u1);
		traceRay(launchParams.traversable, {path.pos, path.dir}, 1e20f,
			RADIANCE_RAY_TYPE, OPTIX_RAY_FLAG_DISABLE_ANYHIT, u0, u1);

		if (sd.miss) {
			handleMiss(sd, path);
			break;
		}
		else{
			handleHit(sd, path);
			if (launchParams.NEE)
				evalDirect(sd, path);
		}
			
		if (!generateScatter(sd, path))
			break;

		// russian roulette
		float u = path.sampler.get1D();
		if (u < launchParams.probRR) break;
		path.throughput /= 1 - launchParams.probRR;
	}

	if (!(path.L < launchParams.clampThreshold))
		path.L = launchParams.clampThreshold;
	// clamp before accumulate?
	path.L = clamp(path.L, vec3f(0), launchParams.clampThreshold);
}

extern "C" __global__ void KRR_RT_RG(Pathtracer)(){
	vec3i launchIndex = optixGetLaunchIndex();
	vec2i pixel = { launchIndex.x, launchIndex.y };

	const int frameID = launchParams.frameID;
	const uint32_t fbIndex = pixel.x + pixel.y * launchParams.fbSize.x;

	Camera& camera = launchParams.camera;
	LCGSampler sampler;		// dont use new(= malloc) here since slow performance 
	sampler.setPixelSample(pixel, frameID);
	// primary ray 
	vec3f rayOrigin = camera.getPosition();
	vec3f rayDir = camera.getRayDir(pixel, launchParams.fbSize, sampler.get2D());
	PathData path = {};

	vec3f color = vec3f(0);

	for (uint i = 0; i < launchParams.spp; i++) {
		PathData path = {};
		path.sampler = &sampler;
		path.pos = rayOrigin; 
		path.dir = rayDir;
		tracePath(path);
		color += path.L;
	}

	color /= launchParams.spp;
	launchParams.colorBuffer[fbIndex] = vec4f(color, 1.0f);
}

KRR_NAMESPACE_END
