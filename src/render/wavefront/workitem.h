#pragma once
#include "common.h"
#include "math/math.h"
#include "raytracing.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

/* Remember to copy these definitions to workitem.soa whenever changing them. */

struct PixelState {
	vec3f L;
	LCGSampler sampler;
	//HaltonSampler sampler;
};

struct RayWorkItem {
	Ray ray;
	vec3f thp;
	uint depth;
	uint pixelId;
};

struct MissRayWorkItem {
	Ray ray;
	vec3f thp;
	uint depth;
	uint pixelId;
};

struct HitLightWorkItem {
	Light light;
	vec3f p;
	vec3f n;
	vec2f uv;
	vec3f wo;
	vec3f thp;
	uint pixelId;
};

struct ShadowRayWorkItem {
	Ray ray;
	float tMax;
	vec3f Li;
	uint pixelId;
};

struct ScatterRayWorkItem {
	vec3f thp;
	ShadingData sd;
	uint depth;
	uint pixelId;
};

#include "workitem_soa.h"
KRR_NAMESPACE_END