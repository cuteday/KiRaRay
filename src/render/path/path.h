#pragma once

#include "math/math.h"
#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

const string shaderProgramNames[] = {
	"Radiance",
	"ShadowRay"
};

struct LaunchParamsPT
{
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
	Vector3f pos;				// ray origin from last scattering event 
	Vector3f dir;				// world space direction of last scatter
	float pdf;					// BxDF sampling pdf from last scatter
	int depth;					// number of vertices along the path
	BSDFType bsdfType;		// the sampled type of the last scatter event
	Sampler sampler;			// rng
	LightSampler lightSampler;	// randomly choosing a light source
};

struct ShadowRayData {
	bool visible = false;
};

KRR_NAMESPACE_END