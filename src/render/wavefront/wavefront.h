#pragma once

#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "device/optix.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

class PixelStateBuffer;

using namespace shader;

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;
	MediumSampleQueue* mediumSampleQueue;
	
	PixelStateBuffer* pixelState;
	rt::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParams;

KRR_NAMESPACE_END