#include <optix_device.h>
#include <optix.h>

#include "math/utils.h"
#include "path.h"
#include "shared.h"
#include "shading.h"

using namespace krr;	// this is needed or nvcc can't recognize the launchParams external var.
using namespace types;

namespace krr
{
	using namespace math;
	using namespace math::utils;
	using namespace shader;
	extern "C" __constant__ LaunchParamsPT launchParams;

	template <typename... Args>
	KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
		float tMax, OptixRayFlags flags, Args &&... payload) {

		optixTrace(traversable, ray.origin, ray.dir,
			0.f, tMax, 0.f,						/* ray time val min max */
			OptixVisibilityMask(255),			/* all visible */
			flags,
			SURFACE_RAY_TYPE, RAY_TYPE_COUNT,	/* ray type and number of types */
			SURFACE_RAY_TYPE,					/* miss SBT index */
			std::forward<Args>(payload)...);
	}

	KRR_DEVICE_FUNCTION void handleHit(const ShadingData& sd, PathData& path) {
		BSDFSample sample = {};

		vec2f u = path.sampler.get2D();
		vec3f wo = sd.toLocal(sd.wo);

		// how to eliminate branches here to improve performance?
		//DiffuseBxDF bsdf;
		//bsdf.setup(sd);
		//sample = bsdf.sample(wo, u);

		sample = BxDF::sampleInternal(sd, u, 0);

		vec3f wi = sd.fromLocal(sample.wi);
		path.ray = { sd.pos + sd.N * 1e-3f, wi };
		path.pdf = sample.pdf;
		path.throughput *= sample.f / max(sample.pdf, 1e-5f);	// to avoid fireflies
	}

	KRR_DEVICE_FUNCTION void handleMiss() {
		// nothing for now...
	}

	extern "C" __global__ void KRR_RT_CH(PathTracer)()
	{
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

	extern "C" __global__ void KRR_RT_AH(PathTracer)()
	{
		return;
	}

	extern "C" __global__ void KRR_RT_MS(PathTracer)()
	{
		// handle envlighting here, just for now...
		ShadingData &sd = *getPRD<ShadingData>();
		vec3f rayDir = optixGetWorldRayDirection();

		LightSample ls = {};
		ls.wi = normalize(rayDir);
		launchParams.envLight.eval(ls);
		sd.emission = ls.Li;
		sd.miss = true;
	}

	KRR_DEVICE_FUNCTION void tracePath(PathData& path) {
		for (uint depth = 0; depth < launchParams.maxDepth; depth++) {
			ShadingData sd = {};
			uint u0, u1;
			packPointer(&sd, u0, u1);
			traceRay(launchParams.traversable, path.ray, 1e20f,
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, u0, u1);
			path.L += path.throughput * sd.emission;

			if (sd.miss) {
				break;
			}
			else {
				handleHit(sd, path);
			}
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

	extern "C" __global__ void KRR_RT_RG(PathTracer)()
	{
		vec3i launchIndex = optixGetLaunchIndex();
		vec2i pixel = { launchIndex.x, launchIndex.y };

		const int frameID = launchParams.frameID;
		const uint32_t fbIndex = pixel.x + pixel.y * launchParams.fbSize.x;

		Camera& camera = launchParams.camera;
		LCGSampler sampler;
		sampler.setPixel(pixel, frameID);
		// primary ray 
		vec3f rayOrigin = camera.getPosition();
		vec3f rayDir = camera.getRayDir(pixel, launchParams.fbSize);
		PathData path = {};

		vec3f color = vec3f(0);

		for (uint i = 0; i < launchParams.spp; i++) {
			PathData path = {};
			path.sampler = sampler;
			path.ray = { rayOrigin, rayDir };
			tracePath(path);
			color += path.L;
		}

		color /= launchParams.spp;
		launchParams.colorBuffer[fbIndex] = vec4f(color, 1.0f);
	}
}
