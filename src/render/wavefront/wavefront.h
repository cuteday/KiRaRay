#pragma once

#include "math/math.h"
#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "device/optix.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	
	//vec4f* frameBuffer;
	Scene::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParams;

KRR_NAMESPACE_END