#pragma once
#include "common.h"

#include "render/materials/bxdf.h"
#include "raytracing.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

using namespace shader;
using namespace rt;

/* Remember to copy these definitions to workitem.soa whenever changing them. */

struct PixelState {
	Color L;
	PCGSampler sampler;
	SampledChannel channel;
};

struct RayWorkItem {
	Ray ray;
	LightSampleContext ctx;
	//float pdf;
	Color thp;
	float pu, pl;
	//Color pu, pl;
	BSDFType bsdfType;
	uint depth;
	uint pixelId;
};

struct MissRayWorkItem {
	Ray ray;
	LightSampleContext ctx;
	//float pdf;
	Color thp;
	Color pu, pl;
	BSDFType bsdfType;
	uint depth;
	uint pixelId;
};

struct HitLightWorkItem {
	Light light;
	LightSampleContext ctx;
	//float pdf;
	Vector3f p;
	Vector3f wo;
	Vector3f n;
	Vector2f uv;
	Color thp;
	Color pu, pl;
	BSDFType bsdfType;
	uint depth;
	uint pixelId;
};

struct ShadowRayWorkItem {
	Ray ray;
	float tMax;
	Color Ld;
	Color pu, pl;
	uint pixelId;
};

struct ScatterRayWorkItem {
	Color thp;
	//Color pu;
	SurfaceInteraction intr;
	uint depth;
	uint pixelId;
};

struct MediumSampleWorkItem {
	Ray ray;
	Color thp;
	Color pu, pl;
	float tMax;
	SurfaceInteraction intr;	// has hit a surface as well...
	BSDFType bsdfType;
	uint depth;
	uint pixelId;
};

struct MediumScatterWorkItem {
	Vector3f p;
	Color thp;
	//Color pu, pl;
	Vector3f wo;
	float time;
	Medium medium;
	PhaseFunction phase;
	uint depth;
	uint pixelId;
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "render/wavefront/workitem_soa.h"
#pragma warning (pop)

KRR_NAMESPACE_END