#include "math/utils.h"
#include "path.h"
#include "render/shared.h"
#include "render/shading.h"
#include "render/raytracing.cuh"
#include "util/hash.h"

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
	Vector2i pixel = optixGetLaunchIndex();
	if (pixel == launchParams.debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}

KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
	
	const Light& light = sd.light;
	Interaction intr(sd.pos, sd.wo, sd.frame.N, sd.uv);
	Color Le = sd.light.L(sd.pos, sd.frame.N, sd.uv, sd.wo);

	if (path.depth == 0) {
		path.L = Le; return;
	}
	
	float weight{ 1 };
	if (launchParams.NEE) {
		LightSampleContext ctx = { path.pos, };	// should use pos of prev interaction
		float bsdfPdf = path.pdf;
		float lightPdf = light.pdfLi(intr, ctx) * path.lightSampler.pdf(light);
		if (!(path.bsdfType & BSDF_SPECULAR)) 
		weight = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
		if (isnan(weight) || isinf(weight)) weight = 1;
	}
	path.L += Le * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void handleMiss(const ShadingData& sd, PathData& path) {
	Vector3f wi = normalize(path.dir);

	LightSampleContext ctx = { sd.pos, sd.frame.N };
	Interaction intr(sd.pos);

	for (InfiniteLight& light : *launchParams.sceneData.infiniteLights) {
		float weight = 1.f;
		if (path.depth && launchParams.NEE) {
			float bsdfPdf = path.pdf;
			float lightPdf = light.pdfLi(intr, ctx) * path.lightSampler.pdf(&light);
			weight = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
			if (isnan(weight) || isinf(weight)) weight = 1;
		}
		path.L += path.throughput * weight * light.Li(wi);
	}
}

KRR_DEVICE_FUNCTION void generateShadowRay(const ShadingData& sd, PathData& path, Ray& shadowRay) {
	Vector3f woLocal = sd.frame.toLocal(sd.wo);

	SampledLight sampledLight = path.lightSampler.sample(path.sampler.get1D());
	Light& light = sampledLight.light;
	LightSample ls			  = light.sampleLi(path.sampler.get2D(), 
									{ sd.pos, sd.frame.N });

	Vector3f wiWorld = normalize(ls.intr.p - sd.pos);
	Vector3f wiLocal = sd.frame.toLocal(wiWorld);

	float lightPdf = sampledLight.pdf * ls.pdf;
	if (lightPdf == 0) return;	// We have sampled on the primitive itself...
	float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
	Color bsdfVal	= BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal[2]);
	float misWeight = evalMIS(launchParams.lightSamples, lightPdf, 1, bsdfPdf);
	if (isnan(misWeight) || isinf(misWeight) || !bsdfVal.any()) return;

	shadowRay	 = sd.getInteraction().spawnRay(ls.intr);
	bool visible = traceShadowRay(launchParams.traversable, shadowRay, 1);
	if (visible) path.L += path.throughput * bsdfVal * misWeight / 
		(launchParams.lightSamples * lightPdf) * ls.L;
}

KRR_DEVICE_FUNCTION void evalDirect(const ShadingData& sd, PathData& path) {
	// Disable NEE on specular surfaces.
	//	BSDFType bsdfType = BxDF::flags(sd, (int) sd.bsdfType);
	//	if (!(bsdfType & BSDF_SMOOTH)) return;	
	for (int i = 0; i < launchParams.lightSamples; i++) {
		Ray shadowRay = {};
		generateShadowRay(sd, path, shadowRay);
	}
}

KRR_DEVICE_FUNCTION bool generateScatterRay(const ShadingData& sd, PathData& path) {
	// how to eliminate branches here to improve performance?
	Vector3f woLocal = sd.frame.toLocal(sd.wo);
	BSDFSample sample = BxDF::sample(sd, woLocal, path.sampler, (int)sd.bsdfType);
	if (sample.pdf == 0 || !any(sample.f)) return false;

	Vector3f wiWorld = sd.frame.toWorld(sample.wi);
	path.bsdfType	 = sample.flags;
	path.pos = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
	path.dir = wiWorld;
	path.pdf = sample.pdf;
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

extern "C" __global__ void KRR_RT_AH(Radiance)() {
	if (alphaKilled(launchParams.sceneData.materials)) optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Radiance)() {
	ShadingData &sd = *getPRD<ShadingData>();
	sd.miss = true;
}

extern "C" __global__ void KRR_RT_AH(ShadowRay)() {
	if (alphaKilled(launchParams.sceneData.materials))
		optixIgnoreIntersection();
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
	for (int &depth = path.depth; true; depth++) {
		// ShadingData is updated in CH shader
		// DO NOT enable OPTIX_RAY_FLAG_DISABLE_ANYHIT if alpha-killing is enabled
		traceRay(launchParams.traversable, { path.pos, path.dir }, KRR_RAY_TMAX,
			RADIANCE_RAY_TYPE, OPTIX_RAY_FLAG_NONE, (void*)&sd);

		if (sd.miss) {			// incorporate emission from envmap, by escaped rays
			handleMiss(sd, path); break;
		}
		else if (sd.light) handleHit(sd, path);

		/* If the path is terminated by this vertex, then NEE should not be evaluated
		 * otherwise the MIS weight of this NEE action will be meaningless. */
		if (depth == launchParams.maxDepth || (depth > 0 &&
			launchParams.probRR >= M_EPSILON && 
			path.sampler.get1D() < launchParams.probRR))
			break;
		path.throughput /= 1 - launchParams.probRR;
		
		if (launchParams.NEE) evalDirect(sd, path);

		if (!generateScatterRay(sd, path)) break;
	}
	// note that clamping also eliminates NaN and INF. 
	//path.L = clamp(path.L, 0.f, launchParams.clampThreshold);
}

extern "C" __global__ void KRR_RT_RG(Pathtracer)(){
	Vector3ui launchIndex = optixGetLaunchIndex();
	Vector2ui pixel		   = {  launchIndex[0], launchIndex[1] };

	const uint frameID = launchParams.frameID;
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

	Color color = Color::Zero();
	for (int i = 0; i < launchParams.spp; i++) {
		path.throughput = Color::Ones();
		path.L			= Color::Zero();
		path.pos = cameraRay.origin;
		path.dir = cameraRay.dir;

		tracePath(path);
		color += path.L;
	}
	color /= float(launchParams.spp);
	launchParams.colorBuffer[fbIndex] = Color4f(color, 1.f);
}

KRR_NAMESPACE_END