#pragma once


#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE = 1, RAY_TYPE_COUNT };

const string shaderProgramNames[RAY_TYPE_COUNT] = {
	"Radiance",
	"ShadowRay"
};

struct LaunchParamsPT {
	uint frameID{ 0 };
	Vector2i fbSize = Vector2i::Zero();
	// per pixel debugging output
	bool debugOutput	= false;
	Vector2i debugPixel = { 960, 540 };
	// path tracing parameters
	bool NEE = false; // enable next event estimation (and multiple importance sampling)
	
	int maxDepth		 = 10;
	float probRR		 = 0.2;
	float clampThreshold = 10000; // clamp max radiance contrib per frame
	int spp				 = 1;
	int lightSamples	 = 1;
	// scene 
	Camera camera;
	LightSampler lightSampler;
	Scene::SceneData sceneData;

	Color4f* colorBuffer{ nullptr };
	OptixTraversableHandle traversable{ 0 };
};

struct PathData {
	Color L{};					// total contribution to the current pixel
	Color throughput;			// maintain the throughput of path
	float pdf;					// BxDF sampling pdf from last scatter
	int depth;					// number of vertices along the path
	BSDFType bsdfType;			// the sampled type of the last scatter event
	Sampler sampler;			// rng
	LightSampler lightSampler;	// randomly choosing a light source
	LightSampleContext ctx;		// last context used for direct light sampling
	Ray ray;					// The last scattered ray
};

struct ShadowRayData {
	bool visible = false;
};

KRR_NAMESPACE_END