#include <optix_device.h>
#include <optix.h>

#include "math/utils.h"
#include "LaunchParams.h"
#include "shared.h"
#include "sampler.h"
#include "path.h"

using namespace krr;	// this is needed or nvcc can't recognize the launchParams external var.

namespace krr
{
	using namespace math;
	using namespace math::utils;
	using namespace shader;
	extern "C" __constant__ LaunchParams optixLaunchParams;

	enum{
		SURFACE_RAY_TYPE = 0,
		RAY_TYPE_COUNT
	};

	template <typename... Args>
	KRR_DEVICE_FUNCTION void traceRay(OptixTraversableHandle traversable, Ray ray,
		float tMax, OptixRayFlags flags, Args &&... payload) {

		optixTrace(traversable, ray.origin, ray.dir,
			1e-7f, tMax, 0.f,					/* ray time val min max */
			OptixVisibilityMask(255),			/* all visible */
			flags,
			SURFACE_RAY_TYPE, RAY_TYPE_COUNT,	/* ray type and number of types */
			SURFACE_RAY_TYPE,					/* miss SBT index */
			std::forward<Args>(payload)...);
	}


	KRR_DEVICE_FUNCTION void handleHit(const ShadingData sd, PathData& path) {
		vec2f r2v = path.sampler.get2D();
		vec3f rayDir = cosineSampleHemisphere(r2v);
		float bsdfPdf = rayDir.z;
		vec3f rayWorldDir = sd.fromLocal(rayDir);
		Ray ray = { sd.pos, rayWorldDir };
		
		path.ray = ray;
		path.pdf = bsdfPdf;
		path.throughput *= 1 / bsdfPdf;
		
		// TODO: direct lighting sampling here
	}

	KRR_DEVICE_FUNCTION void handleMiss() {
		// nothing for now...
	}

	extern "C" __global__ void KRR_RT_CH(PathTracer)()
	{
		vec2f barycentric = optixGetTriangleBarycentrics();
		uint primId = optixGetPrimitiveIndex();
		MeshData& mesh = *(MeshData*)optixGetSbtDataPointer();	

		ShadingData& sd = *getPRD<ShadingData>();
		sd.wi = -normalize(vec3f(optixGetWorldRayDirection()));
		unsigned int hitKind = optixGetHitKind();
		vec3f bc = { 1 - barycentric.x - barycentric.y, barycentric.x, barycentric.y};
		vec3i triangle = mesh.indices[primId];

		// prepare shading data
		sd.pos = bc.x * mesh.vertices[triangle.x] +
			bc.y * mesh.vertices[triangle.y] +
			bc.z * mesh.vertices[triangle.z];

		sd.geoN = normalize(cross(mesh.vertices[triangle.y] - mesh.vertices[triangle.x],
			mesh.vertices[triangle.z] - mesh.vertices[triangle.x]));

		sd.N= normalize(
			bc.x * mesh.normals[triangle.x] + 
			bc.y * mesh.normals[triangle.y] +
			bc.z * mesh.normals[triangle.z]);
		// to do: seems some problem exists with optixIsFrontFaceHit()
		//sd.frontFacing = optixIsFrontFaceHit(hitKind);
		sd.frontFacing = dot(sd.wi, sd.N) > 0.f;
		if (!sd.frontFacing) {
			sd.N = -sd.N;
		}
		// generate a fake tbn frame for now...
		sd.T = getPerpendicular(sd.N);
		sd.B = normalize(cross(sd.N, sd.T));

		float shading = 0.2 + 0.8 * dot(sd.N, sd.wi);
		//sd.emission = vec3f(shading);// *shading;
		sd.diffuse = vec3f(1);
		
		sd.miss = false;
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
		ls.wi = rayDir;
		optixLaunchParams.envLight.eval(ls);
		sd.emission = ls.Li;
		sd.miss = true;
	}

	extern "C" __global__ void KRR_RT_RG(PathTracer)()
	{
		vec3i launchIndex = optixGetLaunchIndex();
		vec2i pixel = { launchIndex.x, launchIndex.y };

		const int frameID = optixLaunchParams.frameID;
		const uint32_t fbIndex = pixel.x + pixel.y * optixLaunchParams.fbSize.x;

		Camera& camera = optixLaunchParams.camera;
		PathData path = {};
		//path.sampler = new LCGSampler();
		path.sampler.setPixel(pixel, frameID);

		// primary ray 
		vec3f rayOrigin = camera.getPosition();
		vec3f rayDir = camera.getRayDir(pixel, optixLaunchParams.fbSize);

		path.L = 0;
		path.throughput = 1;
		path.ray = { rayOrigin, rayDir };

		//if (pixel == vec2i(666, 666) && optixLaunchParams.frameID % 10 == 0)
		//	printf("Testing rng: %f\n", path.sampler.get1D());
		//	printf("Tracing ray at 666, 666: from %f, %f, %f to %f, %f, %f\n", 
		//		rayOrigin.x, rayOrigin.y, rayOrigin.z, rayDir.x, rayDir.y, rayDir.z);

		for (uint depth = 0; depth < optixLaunchParams.maxDepth; depth++) {
			ShadingData sd = {};
			uint u0, u1;
			packPointer(&sd, u0, u1);
			traceRay(optixLaunchParams.traversable, path.ray, 1e10f,
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, u0, u1);
			path.L += path.throughput * sd.emission;
			if (sd.miss) {
				break;
			}
			else {
				handleHit(sd, path);
			}
		}
		
		optixLaunchParams.colorBuffer[fbIndex] = vec4f(path.L, 1.0f);
	}
}
