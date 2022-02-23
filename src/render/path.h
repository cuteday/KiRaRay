#pragma once

#include "math/math.h"
#include "sampler.h"
#include "scene.h"
#include "shared.h"
#include "bsdf.h"

namespace krr{

	using namespace shader;

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		void* data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		void* data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		MeshData data;
	};

	enum {
		RADIANCE_RAY_TYPE = 0,
		SHADOW_RAY_TYPE = 1,
		RAY_TYPE_COUNT
	};

	const string shaderProgramNames[RAY_TYPE_COUNT] = {
		"Radiance",
		"ShadowRay"
	};

	struct LaunchParamsPT
	{
		uint       frameID{ 0 };
		vec4f* colorBuffer;
		vec2i     fbSize = 0;

		bool NEE = false;	// enable NEE + MIS
		uint maxDepth = 20;
		float probRR = 0.15;
		vec3f clampThreshold = 50;
		uint spp = 1;

		vec2i debugPixel = { 666, 666 };

		Camera camera;
		EnvLight envLight;
		Scene::SceneData sceneData;

		OptixTraversableHandle traversable;
	};

	struct PathData {
		// info of this path			
		vec3f L = 0;			// total contribution to the current pixel
		vec3f throughput = 1;	//
		// info of the current sampled scattering ray
		vec3f pos;
		vec3f dir;
		//Ray ray;
		float pdf;			
		// random sample generator
		Sampler sampler;
	};

	struct ShadowRayData {
		bool visible = false;
	};
}