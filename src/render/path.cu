#include "math/utils.h"
#include "path.h"
#include "shared.h"
#include "shading.h"

#include <optix_device.h>

using namespace krr;	// this is needed or nvcc can't recognize the launchParams extern "C"al var.

KRR_NAMESPACE_BEGIN

namespace path {
	constexpr float kMaxDistance = 1e9f;
	constexpr float kShadowEpsilon = 1e-4f;
	constexpr float kSampleEnvLightPdf = 0.2f;
	constexpr float kSampleAreaLightPdf = 1 - kSampleEnvLightPdf;
}

using namespace path;
using namespace math;
using namespace math::utils;
using namespace shader;
using namespace types;

extern "C" __constant__ LaunchParamsPT launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void print(const char* fmt, Args &&... args) {
	if (!launchParams.debugOutput) return;
	vec2i pixel = (vec2i)optixGetLaunchIndex();
	if (pixel == launchParams.debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, Args &&... payload) {

	optixTrace(traversable, ray.origin, ray.dir,
		0.f, tMax, 0.f,						/* ray time val min max */
		OptixVisibilityMask(255),			/* all visible */
		flags,
		rayType, RAY_TYPE_COUNT,			/* ray type and number of types */
		rayType,							/* miss SBT index */
		std::forward<Args>(payload)...);	/* (unpacked pointers to) payloads */
}

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
	float tMax, int rayType, OptixRayFlags flags, void* payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, u0, u1);
}

template <typename... Args>
KRR_DEVICE_FUNCTION bool traceShadowRay(OptixTraversableHandle traversable, 
	Ray ray, float tMax) {
	ShadowRayData sd = { false };
	OptixRayFlags flags = (OptixRayFlags)(OPTIX_RAY_FLAG_DISABLE_ANYHIT);
	uint u0, u1;
	packPointer(&sd, u0, u1);
	traceRay(traversable, ray, tMax, (int)SHADOW_RAY_TYPE,
		flags, u0, u1);
	return sd.visible;
}

KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
	
	const Light& light = sd.light;
	// incorporate the contribution from diffuse surface emission 
	Interaction intr = Interaction(sd.pos, sd.wo, sd.N, sd.uv);
	vec3f Le = sd.light.L(sd.pos, sd.N, sd.uv, sd.wo);

	if (path.depth == 0) {
		// emission at primary vertex
		path.L = Le;
		return;
	}

	float weight = 1.f;
	if (launchParams.NEE) {
		LightSampleContext ctx = { path.pos, };
		float bsdfPdf = path.pdf;
		float lightChoosePdf = path.lightSampler.pdf(light);
		float lightSamplePdf = light.pdfLi(intr, ctx);
		float lightPdf = lightChoosePdf * lightSamplePdf * kSampleAreaLightPdf;
		weight = evalMIS(bsdfPdf, lightPdf);
		print("NEE bsdf sampled diffuse area bsdfPdf: %.5f lightPdf: %.5f weight: %.5f\n",
			bsdfPdf, lightPdf, weight);
	}
	path.L += Le * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void handleMiss(const ShadingData& sd, PathData& path) {
	vec3f wi = normalize(path.dir);
	vec3f Li = launchParams.envLight.eval(wi);

	if (path.depth == 0) {
		// emission at primary vertex
		path.L = Li;
		return;
	}

	float weight = 1.f; 
	if (launchParams.NEE) {
		float bsdfPdf = path.pdf;
		float lightPdf = launchParams.envLight.pdf(wi) * kSampleEnvLightPdf;
		weight = evalMIS(bsdfPdf, lightPdf);
	}
	path.L += Li * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void evalDirect(const ShadingData& sd, PathData& path) {
	// currently evaluate environment map only.
	
	vec2f u = path.sampler.get2D();
	vec3f woLocal = sd.toLocal(sd.wo);

	if (u[0] < kSampleEnvLightPdf) {
		u[0] /= kSampleEnvLightPdf;
		EnvLightSample ls = launchParams.envLight.sample(u);

		vec3f wiLocal = ls.wi;
		vec3f wiWorld = sd.fromLocal(wiLocal);
		if (dot(wiWorld, sd.N) < 0) return;

		float lightPdf = ls.pdf * kSampleEnvLightPdf;
		float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
		vec3f bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal.z);
		float misWeight = evalMIS(lightPdf, bsdfPdf);

		vec3f p = offsetRayOrigin(sd.pos, sd.N, wiWorld);
		Ray shadowRay = { p, wiWorld };
		bool visible = traceShadowRay(launchParams.traversable, shadowRay, kMaxDistance);

		if (visible)
			path.L += path.throughput * bsdfVal * misWeight / lightPdf * ls.L;
	}
	else {
		u[0] = (u[0] - kSampleEnvLightPdf) / kSampleAreaLightPdf;
		SampledLight sampledLight = path.lightSampler.sample(u[0]);
		Light& light = sampledLight.light;
		assert(light);
		LightSample ls = light.sampleLi(u, { sd.pos, sd.N });

		vec3f wiWorld = normalize(ls.intr.p - sd.pos);
		vec3f wiLocal = sd.toLocal(wiWorld);

		float lightPdf = sampledLight.pdf * ls.pdf * kSampleAreaLightPdf;
		float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
		vec3f bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal.z);
		float misWeight = evalMIS(lightPdf, bsdfPdf);

		if (lightPdf == 0 || isnan(lightPdf) || isinf(lightPdf)) return;

		vec3f p = offsetRayOrigin(sd.pos, sd.N, wiWorld);
		vec3f to = ls.intr.offsetRayOrigin(p - ls.intr.p);
		Ray shadowRay = { p, to - p };
		bool visible = traceShadowRay(launchParams.traversable, shadowRay, 1 - kShadowEpsilon);
	
		if (visible)
			path.L += path.throughput * bsdfVal * misWeight / lightPdf * ls.L;
	}
}

KRR_DEVICE_FUNCTION bool generateScatter(const ShadingData& sd, PathData& path) {
	BSDFSample sample = {};

	// how to eliminate branches here to improve performance?
	vec3f woLocal = sd.toLocal(sd.wo);
	sample = BxDF::sample(sd, woLocal, path.sampler, (int)sd.bsdfType);
	if (woLocal.z < 0) {
		print("wi.z: %f\n", sample.wi.z);
		print("Transmit out! F: %f, PDF: %f\n", sample.f, sample.pdf);
	}
	if (sample.pdf == 0 || !any(sample.f)) return false;

	vec3f wiWorld = sd.fromLocal(sample.wi);
	path.pos = offsetRayOrigin(sd.pos, sd.N, wiWorld);
	path.dir = wiWorld;
	path.pdf = max(sample.pdf, 1e-7f);
	path.throughput *= sample.f * fabs(dot(wiWorld, sd.N)) / path.pdf;
	return true;
}

extern "C" __global__ void KRR_RT_CH(Radiance)(){
	HitInfo hitInfo = {};
	HitGroupSBTData* hitData = (HitGroupSBTData*)optixGetSbtDataPointer();
	uint meshId = hitData->meshId;
	vec2f barycentric = (vec2f)optixGetTriangleBarycentrics();
	hitInfo.primitiveId = optixGetPrimitiveIndex();
	hitInfo.mesh = &launchParams.sceneData.meshes[meshId];
	hitInfo.wo = -normalize((vec3f)optixGetWorldRayDirection());
	hitInfo.hitKind = optixGetHitKind();
	hitInfo.barycentric = { 1 - barycentric.x - barycentric.y, barycentric.x, barycentric.y };

	ShadingData& sd = *getPRD<ShadingData>();
	sd.miss = false;
	Material& material = launchParams.sceneData.materials[hitInfo.mesh->materialId];

	prepareShadingData(sd, hitInfo, material);
}

extern "C" __global__ void KRR_RT_AH(Radiance)() {	//skipped
	return;
}

extern "C" __global__ void KRR_RT_MS(Radiance)() {
	ShadingData &sd = *getPRD<ShadingData>();
	sd.miss = true;
}

extern "C" __global__ void KRR_RT_CH(ShadowRay)() {	//skipped
	return;
}

extern "C" __global__ void KRR_RT_AH(ShadowRay)() {	//skipped
	return;
}

extern "C" __global__ void KRR_RT_MS(ShadowRay)() {
	ShadowRayData& sd = *getPRD<ShadowRayData>();
	sd.visible = true;
}

KRR_DEVICE_FUNCTION void tracePath(PathData& path) {
	ShadingData sd = {};

	// an alternate version of main loop
	for (int &depth = path.depth; depth < launchParams.maxDepth; depth++) {
		// ShadingData is updated in CH shader
		traceRay(launchParams.traversable, { path.pos, path.dir }, 1e20f,
			RADIANCE_RAY_TYPE, OPTIX_RAY_FLAG_DISABLE_ANYHIT, (void*)&sd);

		if (sd.miss) {			// incorporate emission from envmap, by escaped rays
			handleMiss(sd, path);
			break;
		}
		else if (sd.light){		// incorporate emission from surface area lights
			handleHit(sd, path);
		}

		if (launchParams.NEE) {
			evalDirect(sd, path);
		}
		if (launchParams.RR) {
			float u = path.sampler.get1D();
			if (u < launchParams.probRR) break;
			path.throughput /= 1 - launchParams.probRR;
		}
		if (!generateScatter(sd, path)) break;
	}
	path.L = clamp(path.L, vec3f(0), launchParams.clampThreshold);
}

extern "C" __global__ void KRR_RT_RG(Pathtracer)(){
	vec3i launchIndex = (vec3i)optixGetLaunchIndex();
	vec2i pixel = { launchIndex.x, launchIndex.y };

	const int frameID = launchParams.frameID;
	const uint32_t fbIndex = pixel.x + pixel.y * launchParams.fbSize.x;

	Camera& camera = launchParams.camera;
	LCGSampler sampler;
	sampler.setPixelSample(pixel, frameID);

	// primary ray 
	vec3f rayOrigin = camera.getPosition();
	vec3f rayDir = camera.getRayDir(pixel, launchParams.fbSize, sampler.get2D());
	
	PathData path = {};
	path.lightSampler = &launchParams.sceneData.lightSampler;
	path.sampler = &sampler;

	vec3f color = 0;
	for (int i = 0; i < launchParams.spp; i++) {
		path.throughput = 1;
		path.L = 0;
		path.pos = rayOrigin;
		path.dir = rayDir;

		tracePath(path);
		color += path.L;
	}
	color /= launchParams.spp;
	//print("Final radiance: %.4f %.4f %.4f\n", color.x, color.y, color.z);
	assert(!isnan(luminance(color)));
	launchParams.colorBuffer[fbIndex] = vec4f(color, 1.0f);
}

KRR_NAMESPACE_END