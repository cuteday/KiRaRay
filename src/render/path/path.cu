#include "math/utils.h"
#include "path.h"
#include "render/shared.h"
#include "render/shading.cuh"
#include "render/raytracing.cuh"

#include <optix_device.h>

using namespace krr;	// this is needed or nvcc cannot recognize the launchParams extern "C" var.
KRR_NAMESPACE_BEGIN

using namespace math;
using namespace math::utils;
using namespace shader;
using namespace types;

extern "C" __constant__ LaunchParamsPT launchParams;

KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
								  float tMax, int rayType, OptixRayFlags flags,
								  void *payload) {
	uint u0, u1;
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, u0, u1);
}

template <typename... Args>
KRR_DEVICE_FUNCTION bool traceShadowRay(OptixTraversableHandle traversable,
										Ray ray, float tMax) {
	ShadowRayData sd	= { false };
	OptixRayFlags flags = (OptixRayFlags) (OPTIX_RAY_FLAG_DISABLE_ANYHIT);
	uint u0, u1;
	packPointer(&sd, u0, u1);
	traceRay(traversable, ray, tMax, (int) SHADOW_RAY_TYPE, flags, u0, u1);
	return sd.visible;
}

template <typename... Args>
KRR_DEVICE_FUNCTION void print(const char* fmt, Args &&... args) {
	if (!launchParams.debugOutput) return;
	vec2i pixel = (vec2i) optixGetLaunchIndex();
	if (pixel == launchParams.debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}

KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
	
	const Light& light = sd.light;
	// incorporate the contribution from diffuse surface emission 
	Interaction intr = Interaction(sd.pos, sd.wo, sd.frame.N, sd.uv);
	vec3f Le = sd.light.L(sd.pos, sd.frame.N, sd.uv, sd.wo);

	if (path.depth == 0) {
		// emission at primary vertex
		path.L = Le;
		return;
	}

	float weight = 1.f;
	if (launchParams.NEE) {
		LightSampleContext ctx = { path.pos, };	// should use pos of prev interaction
		float bsdfPdf = path.pdf;
		float lightPdf = light.pdfLi(intr, ctx) * path.lightSampler.pdf(light);
		if (launchParams.MIS) weight = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
		print("NEE bsdf sampled diffuse area bsdfPdf: %.5f lightPdf: %.5f weight: %.5f\n",
			bsdfPdf, lightPdf, weight);
	}
	path.L += Le * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void handleMiss(const ShadingData& sd, PathData& path) {
	vec3f wi = normalize(path.dir);

	LightSampleContext ctx = { sd.pos, sd.frame.N };
	Interaction intr(sd.pos);

	for (InfiniteLight& light : *launchParams.sceneData.infiniteLights) {
		float weight = 1.f;
		if (path.depth&&launchParams.NEE) {
			float bsdfPdf = path.pdf;
			float lightPdf = light.pdfLi(intr, ctx) * path.lightSampler.pdf(&light);
			if (launchParams.MIS) weight = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
		}
		path.L += path.throughput * weight * light.Li(wi);
	}
}

KRR_DEVICE_FUNCTION void generateShadowRay(const ShadingData& sd, PathData& path, Ray& shadowRay) {

	vec2f u = path.sampler.get2D();
	vec3f woLocal = sd.frame.toLocal(sd.wo);

	SampledLight sampledLight = path.lightSampler.sample(u[0]);
	Light& light = sampledLight.light;
	LightSample ls = light.sampleLi(u, { sd.pos, sd.frame.N });

	vec3f wiWorld = normalize(ls.intr.p - sd.pos);
	vec3f wiLocal = sd.frame.toLocal(wiWorld);

	float lightPdf = sampledLight.pdf * ls.pdf;
	float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
	vec3f bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal[2]);
	float misWeight = 1;

	if (launchParams.MIS) misWeight = evalMIS(launchParams.lightSamples, lightPdf, 1, bsdfPdf);
	//if (isnan(misWeight)) 
	// TODO: check why ls.pdf (shape_sample.pdf) can potentially be zero.
	//	printf("nee misWeight %f lightPdf %f bsdfPdf %f lightSelect %f lightSample %f\n",
	//		misWeight, lightPdf, bsdfPdf, sampledLight.pdf, ls.pdf);
	vec3f p = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
	vec3f to = ls.intr.offsetRayOrigin(p - ls.intr.p);

	shadowRay = { p, to - p };
	bool visible = traceShadowRay(launchParams.traversable, shadowRay, 1);
	if (visible) path.L += path.throughput * bsdfVal * misWeight / (launchParams.lightSamples * lightPdf) * ls.L;
}

KRR_DEVICE_FUNCTION void evalDirect(const ShadingData& sd, PathData& path) {
	for (int i = 0; i < launchParams.lightSamples; i++) {
		Ray shadowRay = {};
		generateShadowRay(sd, path, shadowRay);
	}
}

KRR_DEVICE_FUNCTION bool generateScatterRay(const ShadingData& sd, PathData& path) {
	// how to eliminate branches here to improve performance?
	vec3f woLocal = sd.frame.toLocal(sd.wo);
	BSDFSample sample = BxDF::sample(sd, woLocal, path.sampler, (int)sd.bsdfType);
	if (sample.pdf == 0 || !any(sample.f)) return false;

	vec3f wiWorld = sd.frame.toWorld(sample.wi);
	path.pos = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
	path.dir = wiWorld;
	path.pdf = max(sample.pdf, 1e-7f);
	path.throughput *= sample.f * fabs(sample.wi[2]) / path.pdf;
	return true;
}

extern "C" __global__ void KRR_RT_CH(Radiance)(){
	HitInfo hitInfo = getHitInfo();
	ShadingData& sd = *getPRD<ShadingData>();
	sd.miss = false;
	Material& material = (*launchParams.sceneData.materials)[hitInfo.mesh->materialId];
	prepareShadingData(sd, hitInfo, material);
}

extern "C" __global__ void KRR_RT_MS(Radiance)() {
	ShadingData &sd = *getPRD<ShadingData>();
	sd.miss = true;
}

extern "C" __global__ void KRR_RT_CH(ShadowRay)() {	//skipped
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
		traceRay(launchParams.traversable, { path.pos, path.dir }, KRR_RAY_TMAX,
			RADIANCE_RAY_TYPE, OPTIX_RAY_FLAG_DISABLE_ANYHIT, (void*)&sd);

		if (sd.miss) {			// incorporate emission from envmap, by escaped rays
			handleMiss(sd, path); break;
		}
		else if (sd.light) handleHit(sd, path);
		
		if (launchParams.NEE) evalDirect(sd, path);
		
		if (launchParams.RR) {
			float u = path.sampler.get1D();
			if (u < launchParams.probRR) break;
			path.throughput /= 1 - launchParams.probRR;
		}

		if (!generateScatterRay(sd, path)) break;
	}
	// note that clamping also eliminates NaN and INF. 
	path.L = clamp(path.L, 0.f, launchParams.clampThreshold);
}

extern "C" __global__ void KRR_RT_RG(Pathtracer)(){
	vec3i launchIndex = (vec3i)optixGetLaunchIndex();
	vec2i pixel = { launchIndex[0], launchIndex[1] };

	const int frameID = launchParams.frameID;
	const uint32_t fbIndex = pixel[0] + pixel[1] * launchParams.fbSize[0];

	Camera& camera = launchParams.camera;
	PCGSampler sampler;
	sampler.setPixelSample(pixel, frameID);
	sampler.advance(fbIndex * 256);

	// primary ray 
	Ray cameraRay = camera.getRay(pixel, launchParams.fbSize, &sampler);
	
	PathData path = {};
	path.lightSampler = launchParams.sceneData.lightSampler;
	path.sampler = &sampler;

	vec3f color = vec3f::Zero();
	for (int i = 0; i < launchParams.spp; i++) {
		path.throughput = vec3f::Ones();
		path.L			= vec3f::Zero();
		path.pos = cameraRay.origin;
		path.dir = cameraRay.dir;

		tracePath(path);
		color += path.L;
	}
	color /= launchParams.spp;
	assert(!isnan(luminance(color)));
	launchParams.colorBuffer[fbIndex] = vec4f(color, 1.0f);
}

KRR_NAMESPACE_END