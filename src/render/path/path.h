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
	vec2i fbSize = vec2i::Zero();
	// per pixel debugging output
	bool debugOutput = false;
	vec2i debugPixel = { 960, 540 };
	// path tracing parameters
	bool RR = true;				// enable russian roulette
	bool NEE = false;			// enable next event estimation
	bool MIS = true;			// enable multiple importance sampling. if disable but NEE enabled, "some types" of the lights (i.e. area lights and env lights) will be counted twice.
	int maxDepth = 10;
	float probRR = 0.15;
	float clampThreshold = 100; // clamp max radiance contrib per frame
	int spp = 1;
	int lightSamples = 1;
	// scene 
	Camera camera;
	LightSampler lightSampler;
	Scene::SceneData sceneData;

	vec4f* colorBuffer{ nullptr };
	OptixTraversableHandle traversable{ 0 };
};

struct PathData {
	vec3f L{};					// total contribution to the current pixel
	vec3f throughput;			// maintain the throughput of path
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

KRR_NAMESPACE_END