#pragma once

#include "math/math.h"
#include "sampler.h"
#include "lightsampler.h"
#include "scene.h"
#include "shared.h"
#include "bsdf.h"

namespace krr{

	using namespace shader;

	struct HitGroupSBTData {
		uint meshId;
	};

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
		HitGroupSBTData data;
	};

	enum {
		RADIANCE_RAY_TYPE = 0,
		SHADOW_RAY_TYPE = 1,
		RAY_TYPE_COUNT
	};

	const string shaderProgramNames[] = {
		"Radiance",
		"ShadowRay"
	};

	struct LaunchParamsPT
	{
		uint frameID{ 0 };
		vec2i fbSize = 0;
		// debugging output
		bool debugOutput = false;
		vec2i debugPixel = { 960, 540 };
		// path tracing parameters
		bool NEE = false;			// enable next event estimation
		bool MIS = false;			// enable multiple importance sample
		uint maxDepth = 20;
		float probRR = 0.15;
		vec3f clampThreshold = 5000;
		uint spp = 1;
		// scene 
		Camera camera;
		EnvLight envLight;
		LightSampler lightSampler;
		Scene::SceneData sceneData;

		vec4f* colorBuffer{ nullptr };
		OptixTraversableHandle traversable{ 0 };
	};

	struct PathData {
		vec3f L = 0;				// total contribution to the current pixel
		vec3f throughput;		// maintain the throughput of path
		vec3f pos;					// ray origin from last scattering event 
		vec3f dir;					// world space direction of last scatter
		float pdf;					// BxDF sampling pdf from last scatter
		int depth;					// number of vertices along the path
		Sampler sampler;			// rng
		LightSampler lightSampler;	// randomly choosing a light source
	};

	struct ShadowRayData {
		bool visible = false;
	};
}