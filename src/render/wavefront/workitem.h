#pragma once
#include "common.h"
#include "math/math.h"
#include "raytracing.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

/* Remember to copy these definitions to workitem.soa whenever changing them. */

struct PixelState {
	Color L;
	PCGSampler sampler;
};

struct RayWorkItem {
	Ray ray;
	LightSampleContext ctx;
	float pdf;
	Color thp;
	uint depth;
	uint pixelId;
};

struct MissRayWorkItem {
	Ray ray;
	LightSampleContext ctx;
	float pdf;
	Color thp;
	uint depth;
	uint pixelId;
};

struct HitLightWorkItem {
	Light light;
	LightSampleContext ctx;
	float pdf;
	Vector3f p;
	Vector3f wo;
	Vector3f n;
	Vector2f uv;
	Color thp;
	uint depth;
	uint pixelId;
};

struct ShadowRayWorkItem {
	Ray ray;
	float tMax;
	Color Li;
	Color a;
	uint pixelId;
};

struct ScatterRayWorkItem {
	Color thp;
	ShadingData sd;
	uint depth;
	uint pixelId;
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "workitem_soa.h"
#pragma warning (pop)

KRR_NAMESPACE_END