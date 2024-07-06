#include "util/math_utils.h"
#include "path.h"
#include "render/shared.h"
#include "render/shading.h"
#include "util/hash.h"

#include <optix_device.h>

NAMESPACE_BEGIN(krr)

using namespace rt;

extern "C" __constant__ LaunchParameters<MegakernelPathTracer> launchParams;

template <typename... Args>
KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray, float tMax,
								  int rayType, OptixRayFlags flags, Args &&...payload) {
	optixTrace(traversable, ray.origin, ray.dir, 0.f, tMax, ray.time, /* ray time val min max */
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

KRR_DEVICE_FUNCTION void handleHit(PathData& path) {
	const SurfaceInteraction &intr = path.intr;
	const rt::Light &light		   = intr.light;
	Spectrum Le			   = light.L(intr.p, intr.n, intr.uv, intr.wo, path.lambda);

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

KRR_DEVICE_FUNCTION void handleMiss(PathData& path) {
	const SurfaceInteraction &intr = path.intr;
	for (const rt::InfiniteLight &light : launchParams.sceneData.infiniteLights) {
		float weight{ 1 };
		if (launchParams.NEE && path.depth > 0 && !(path.bsdfType & BSDF_SPECULAR)) {
			float bsdfPdf  = path.pdf;
			float lightPdf = light.pdfLi({}, path.ctx) * path.lightSampler.pdf(&light);
			weight		   = evalMIS(1, bsdfPdf, launchParams.lightSamples, lightPdf);
			if (isnan(weight) || isinf(weight))
				weight = 1;
		}
		path.L += path.throughput * weight * light.Li(path.ray.dir, path.lambda);
	}
}

KRR_DEVICE_FUNCTION void generateShadowRay(PathData& path) {
	const SurfaceInteraction &intr = path.intr;
	Vector3f woLocal = intr.toLocal(intr.wo);

	SampledLight sampledLight = path.lightSampler.sample(path.sampler.get1D());
	Light light				  = sampledLight.light;
	LightSample ls			  = light.sampleLi(path.sampler.get2D(), { intr.p, intr.n }, path.lambda);

	Vector3f wiWorld = normalize(ls.intr.p - intr.p);
	Vector3f wiLocal = intr.toLocal(wiWorld);

	float lightPdf = sampledLight.pdf * ls.pdf;
	if (lightPdf == 0)
		return; // We have sampled on the primitive itself...
	float bsdfPdf = light.isDeltaLight() ? 0 : BxDF ::pdf(intr, woLocal, wiLocal);
	Spectrum bsdfVal =
		BxDF::f(intr, woLocal, wiLocal) * fabs(wiLocal[2]);
	float misWeight = evalMIS(launchParams.lightSamples, lightPdf, 1, bsdfPdf);
	if (isnan(misWeight) || isinf(misWeight) || !bsdfVal.any()) return;

	Ray shadowRay = intr.spawnRayTo(ls.intr);
	if (traceShadowRay(launchParams.traversable, shadowRay, 1))
		path.L +=
			path.throughput * bsdfVal * misWeight / (launchParams.lightSamples * lightPdf) * ls.L;
}

KRR_DEVICE_FUNCTION void evalDirect(PathData& path) {
	BSDFType bsdfType			   = path.intr.getBsdfType();
	if (bsdfType & BSDF_SMOOTH) {	/* Disable NEE on specular surfaces. */
		for (int i = 0; i < launchParams.lightSamples; i++) 
			generateShadowRay(path);
	}
}

KRR_DEVICE_FUNCTION bool generateScatterRay(PathData& path) {
	const SurfaceInteraction &intr = path.intr;
	Vector3f woLocal			   = intr.toLocal(intr.wo);
	BSDFSample sample = BxDF::sample(intr, woLocal, path.sampler);
	if (sample.pdf == 0 || !any(sample.f)) return false;

	Vector3f wiWorld = intr.toWorld(sample.wi);
	path.bsdfType	 = sample.flags;
	path.ray		 = intr.spawnRayTowards(wiWorld);
	path.ctx		 = { intr.p, intr.n };
	path.pdf		 = sample.pdf;
	path.throughput *= sample.f * fabs(sample.wi[2]) / sample.pdf;
	return path.throughput.any();
}

KRR_RT_KERNEL KRR_RT_CH(Radiance)(){
	HitInfo hitInfo			 = getHitInfo();
	PathData *path			 = getPRD<PathData>();
	SurfaceInteraction &intr = path->intr;	
	prepareSurfaceInteraction(intr, hitInfo, path->ray, path->lambda);
}

KRR_RT_KERNEL KRR_RT_AH(Radiance)() {
	if (alphaKilled(getHitInfo())) optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_MS(Radiance)() {
	optixSetPayload_2(1);
}

KRR_RT_KERNEL KRR_RT_AH(ShadowRay)() {
	if (alphaKilled(getHitInfo())) optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_CH(ShadowRay)() {} /* skipped */

KRR_RT_KERNEL KRR_RT_MS(ShadowRay)() { optixSetPayload_0(1); }

KRR_DEVICE_FUNCTION void tracePath(PathData& path) {
	for (int &depth = path.depth; true; depth++) {
		if(!traceRay(launchParams.traversable, path.ray, M_FLOAT_INF, RADIANCE_RAY_TYPE,
					  OPTIX_RAY_FLAG_NONE, (void *) &path)) {
			handleMiss(path);
			break;
		}
		
		if (path.intr.light) handleHit(path);

		/* If the path is terminated by this vertex, then NEE should not be evaluated
		 * otherwise the MIS weight of this NEE action will be meaningless. */
		if (depth == launchParams.maxDepth || 
			(launchParams.probRR < 1.f && path.sampler.get1D() > launchParams.probRR))
			break;
		path.throughput /= launchParams.probRR;
		
		if (launchParams.NEE) evalDirect(path);
		if (!generateScatterRay(path)) break;
	}
}

KRR_RT_KERNEL KRR_RT_RG(Pathtracer)(){
	Vector3ui launchIndex = optixGetLaunchIndex();
	Vector2ui pixel		  = {launchIndex[0], launchIndex[1]};

	const uint frameID = launchParams.frameID;
	const uint32_t fbIndex = pixel[0] + pixel[1] * launchParams.fbSize[0];
	
	rt::CameraData camera = launchParams.camera;
	PCGSampler sampler;
	sampler.setPixelSample(pixel, frameID);
	sampler.advance(fbIndex * 256);

	PathData path	  = {};
	path.lightSampler = launchParams.sceneData.lightSampler;
	path.sampler	  = &sampler;

	RGB color = RGB::Zero();
	for (int i = 0; i < launchParams.spp; i++) {
		path.throughput = Spectrum::Ones();
		path.L			= Spectrum::Zero();
		path.ray		= camera.getRay(pixel, launchParams.fbSize, path.sampler);
		path.lambda		= SampledWavelengths::sampleUniform(sampler.get1D());

		tracePath(path);
		color += path.L.toRGB(path.lambda, *launchParams.colorSpace);
	}

	launchParams.colorBuffer.write(RGBA(color, 1.f), fbIndex);
}

NAMESPACE_END(krr)