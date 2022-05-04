#pragma once
#include "common.h"
#include "math/math.h"
#include "raytracing.h"

KRR_NAMESPACE_BEGIN

struct RayWorkItem {
	Ray ray;
	uint depth;
	vec3f thp;
	uint pixelId;
};

struct MissRayWorkItem {
	Ray ray;
	uint depth;
	vec3f thp;
	uint pixelId;
};

struct HitLightWorkItem {
	Light light;
	vec3f thp;
	uint pixelId;
};

struct ShadowRayWorkItem {
	Ray ray;
	float tMax;
	vec3f Li;
	uint pixelId;
};

#include "workitem_soa.h"
KRR_NAMESPACE_END