#pragma once
#include "common.h"
#include "math/math.h"
#include "raytracing.h"

KRR_NAMESPACE_BEGIN

struct RayWorkItem {
    Ray ray;
    uint depth;
    uint pixelId;
};

struct MissRayWorkItem {
    Ray ray;
    uint depth;
    uint pixelId;
    vec3f thp;
};

struct HitLightWorkItem {
    Light light;
    vec3f thp;
    uint pixelId;
};

#include "workitem_soa.h"
KRR_NAMESPACE_END