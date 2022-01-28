#include <optix_device.h>
#include <optix.h>
//#include <optix_stubs.h>
//#include <optix_host.h>
//#include <optix_function_table.h>


#include "LaunchParams.h"
#include "shared.h"

using namespace krr;

namespace krr
{
	using namespace shader;
	using namespace math;

	// upload updated parameters each rt launch
	extern "C" __constant__ LaunchParams optixLaunchParams;

	enum
	{
		SURFACE_RAY_TYPE = 0,
		RAY_TYPE_COUNT
	};

	extern "C" __global__ void KRR_RT_CH(PathTracer)()
	{
		vec3f &prd = *getPRD<vec3f>();
		int prim_id = optixGetPrimitiveIndex();
		MeshSBTData &mesh = *(MeshSBTData *)optixGetSbtDataPointer();

		vec3i triangle = mesh.indices[prim_id];
		vec3f normal = normalize(mesh.normals[triangle.x] + mesh.normals[triangle.y] + mesh.normals[triangle.z]);
		vec3f view = optixGetWorldRayDirection();
		float shading = clamp(fabs(dot(normal, -view)), 0.f, 1.f);

		prd = vec3f(shading);// *shading;
	}

	extern "C" __global__ void KRR_RT_AH(PathTracer)()
	{
	}

	extern "C" __global__ void KRR_RT_MS(PathTracer)()
	{
		*getPRD<vec3f>() = vec3f(0.9f);
	}

	extern "C" __global__ void KRR_RT_RG(PathTracer)()
	{
		vec3i pixelID = optixGetLaunchIndex();
		vec2i pixel = {pixelID.x, pixelID.y};

		const int frameID = optixLaunchParams.frameID;
		const uint32_t fbIndex = pixel.x + pixel.y * optixLaunchParams.fbSize.x;

		Camera& camera = optixLaunchParams.camera;
		vec3f rayOrigin = camera.getPosition();
		vec3f rayDir = camera.getRayDir(pixel, optixLaunchParams.fbSize);

		//if (pixel == vec2i(666, 666) && optixLaunchParams.frameID % 100 == 0)
		//	printf("Tracing ray at 666, 666: from %f, %f, %f to %f, %f, %f\n", 
		//		rayOrigin.x, rayOrigin.y, rayOrigin.z, rayDir.x, rayDir.y, rayDir.z);

		vec3f prd = vec3f(0);
		uint u0, u1;
		packPointer(&prd, u0, u1);
		optixTrace(optixLaunchParams.traversable,
		 		   rayOrigin,
		 		   rayDir,
		 		   0.f,
		 		   1e10f,
		 		   0.f,
		 		   OptixVisibilityMask(255),
		 		   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		 		   SURFACE_RAY_TYPE,
		 		   RAY_TYPE_COUNT,
		 		   SURFACE_RAY_TYPE,
		 		   u0, u1);		// per ray data pointer (32bits each)

		vec3f color = vec3f(1);
		color *= prd;
		//color = (float)pixel.x / optixLaunchParams.fbSize.x * pixel.y / optixLaunchParams.fbSize.y * vec3f(1.0);
		optixLaunchParams.colorBuffer[fbIndex] = vec4f(vec3f(color), 1.0f);
	}

}
