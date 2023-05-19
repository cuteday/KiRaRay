#include "util/math_utils.h"
#include "path.h"
#include "render/shared.h"
#include "render/shading.h"
#include "util/hash.h"

#include <optix_device.h>

using namespace krr;	// this is needed or nvcc cannot recognize the launchParams extern "C" var.
KRR_NAMESPACE_BEGIN

using namespace utils;
using namespace shader;
using namespace types;

extern "C" __constant__ LaunchParamsPT launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {

	optixTrace(traversable, ray.origin, ray.dir, 0.f, tMax, 0.f, /* ray time val min max */
			   OptixVisibilityMask(255),						 /* all visible */
			   flags, rayType, RAY_TYPE_COUNT,					 /* ray type and number of types */
			   rayType,											 /* miss SBT index */
			   std::forward<Args>(payload)...); /* (unpacked pointers to) payloads */
}

/* @returns: whether this ray is missed */
KRR_DEVICE_FUNCTION bool traceRay(OptixTraversableHandle traversable, Ray ray,
								  float tMax, int rayType, OptixRayFlags flags,
								  void *payload) {
	uint u0, u1;
	uint miss{ 0 };
	packPointer(payload, u0, u1);
	traceRay(traversable, ray, tMax, rayType, flags, u0, u1, miss);
	return !miss;
}

template <typename... Args>
KRR_DEVICE_FUNCTION bool traceShadowRay(OptixTraversableHandle traversable,
										Ray ray, float tMax) {
	OptixRayFlags flags =
		(OptixRayFlags) (OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
	uint32_t miss{ 0 };
	traceRay(traversable, ray, tMax, (int) SHADOW_RAY_TYPE, flags, miss);
	return miss;
}

template <typename... Args>
KRR_DEVICE_FUNCTION void print(const char* fmt, Args &&... args) {
	if (!launchParams.debugOutput) return;
	Vector2i pixel = optixGetLaunchIndex();
	if (pixel == launchParams.debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}

KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
	const Light &light = sd.light;
	Interaction intr   = sd.getInteraction();
	Color Le		   = light.L(intr.p, intr.n, intr.uv, intr.wo);

	float weight{ 1 };
	if (launchParams.NEE && path.depth > 0) {
		float bsdfPdf		   = path.pdf;
		float lightPdf		   = light.pdfLi(intr, path.ctx) * path.lightSampler.pdf(light);
		if (!(path.bsdfType & BSDF_SPECULAR))
			weight = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
		if (isnan(weight) || isinf(weight))
			weight = 1;
	}
	path.L += Le * weight * path.throughput;
}

KRR_DEVICE_FUNCTION void handleMiss(const ShadingData& sd, PathData& path) {
	for (InfiniteLight &light : *launchParams.sceneData.infiniteLights) {
		float weight{ 1 };
		if (launchParams.NEE && path.depth > 0 && !(path.bsdfType & BSDF_SPECULAR)) {
			float bsdfPdf  = path.pdf;
			float lightPdf = light.pdfLi({}, path.ctx) * path.lightSampler.pdf(&light);
			weight		   = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
			if (isnan(weight) || isinf(weight))
				weight = 1;
		}
		path.L += path.throughput * weight * light.Li(path.ray.dir);
	}
}

KRR_DEVICE_FUNCTION void generateShadowRay(const ShadingData& sd, PathData& path) {
	Vector3f woLocal = sd.frame.toLocal(sd.wo);

	SampledLight sampledLight = path.lightSampler.sample(path.sampler.get1D());
	Light &light			  = sampledLight.light;
	LightSample ls			  = light.sampleLi(path.sampler.get2D(), { sd.pos, sd.frame.N });

	Vector3f wiWorld = normalize(ls.intr.p - sd.pos);
	Vector3f wiLocal = sd.frame.toLocal(wiWorld);

	float lightPdf = sampledLight.pdf * ls.pdf;
	if (lightPdf == 0)
		return; // We have sampled on the primitive itself...
	float bsdfPdf	= BxDF::pdf(sd, woLocal, wiLocal, (int) sd.bsdfType);
	Color bsdfVal	= BxDF::f(sd, woLocal, wiLocal, (int) sd.bsdfType) * fabs(wiLocal[2]);
	float misWeight = evalMIS(launchParams.lightSamples, lightPdf, 1, bsdfPdf);
	if (isnan(misWeight) || isinf(misWeight) || !bsdfVal.any())
		return;

	Ray shadowRay = sd.getInteraction().spawnRay(ls.intr);
	if (traceShadowRay(launchParams.traversable, shadowRay, 1))
		path.L +=
			path.throughput * bsdfVal * misWeight / (launchParams.lightSamples * lightPdf) * ls.L;
}

KRR_DEVICE_FUNCTION void evalDirect(const ShadingData& sd, PathData& path) {
	BSDFType bsdfType = sd.getBsdfType();
	if (bsdfType & BSDF_SMOOTH) {	/* Disable NEE on specular surfaces. */
		for (int i = 0; i < launchParams.lightSamples; i++) 
			generateShadowRay(sd, path);
	}
}

KRR_DEVICE_FUNCTION bool generateScatterRay(const ShadingData& sd, PathData& path) {
	// how to eliminate branches here to improve performance?
	Vector3f woLocal = sd.frame.toLocal(sd.wo);
	BSDFSample sample = BxDF::sample(sd, woLocal, path.sampler, (int)sd.bsdfType);
	if (sample.pdf == 0 || !any(sample.f)) return false;

	Vector3f wiWorld = sd.frame.toWorld(sample.wi);
	Vector3f po		 = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
	path.bsdfType	 = sample.flags;
	path.ray		 = { po, wiWorld };
	path.ctx		 = { sd.pos, sd.frame.N };
	path.pdf		 = sample.pdf;
	path.throughput *= sample.f * fabs(sample.wi[2]) / sample.pdf;
	return path.throughput.any();
}

extern "C" __global__ void KRR_RT_CH(Radiance)(){
	HitInfo hitInfo	   = getHitInfo();
	ShadingData &sd	   = *getPRD<ShadingData>();
	const rt::MaterialData &material = hitInfo.instance->getMaterial();
	prepareShadingData(sd, hitInfo, material);
}

extern "C" __global__ void KRR_RT_AH(Radiance)() {
	if (alphaKilled()) 
		optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_MS(Radiance)() {
	optixSetPayload_2(1);
}

extern "C" __global__ void KRR_RT_AH(ShadowRay)() {
	if (alphaKilled())
		optixIgnoreIntersection();
}

extern "C" __global__ void KRR_RT_CH(ShadowRay)() {} /* skipped */

extern "C" __global__ void KRR_RT_MS(ShadowRay)() { optixSetPayload_0(1); }

KRR_DEVICE_FUNCTION void tracePath(PathData& path) {
	ShadingData sd = {};

	for (int &depth = path.depth; true; depth++) {
		if(!traceRay(launchParams.traversable, path.ray, KRR_RAY_TMAX, RADIANCE_RAY_TYPE,
					  OPTIX_RAY_FLAG_NONE, (void *) &sd)) {
			handleMiss(sd, path);
			break;
		}
		
		if (sd.light) handleHit(sd, path);

		/* If the path is terminated by this vertex, then NEE should not be evaluated
		 * otherwise the MIS weight of this NEE action will be meaningless. */
		if (depth == launchParams.maxDepth || 
			(launchParams.probRR < 1.f && path.sampler.get1D() > launchParams.probRR))
			break;
		path.throughput /= launchParams.probRR;
		
		if (launchParams.NEE) evalDirect(sd, path);

		if (!generateScatterRay(sd, path)) break;
	}
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

	PathData path	  = {};
	path.lightSampler = launchParams.sceneData.lightSampler;
	path.sampler	  = &sampler;

	Color color = Color::Zero();
	for (int i = 0; i < launchParams.spp; i++) {
		path.throughput = Color::Ones();
		path.L			= Color::Zero();
		path.ray		= camera.getRay(pixel, launchParams.fbSize, path.sampler);

		tracePath(path);
		color += path.L;
	}
	color /= float(launchParams.spp);
	launchParams.colorBuffer.write(Color4f(color, 1.f), fbIndex);
}

KRR_NAMESPACE_END