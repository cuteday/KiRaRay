#pragma once
#include "render/wavefront/wavefront.h"
#include "guideditem.h"

KRR_NAMESPACE_BEGIN

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;

	bool enableTraining;
	PixelStateBuffer* pixelState;
	rt::SceneData sceneData;
	GuidedPathStateBuffer* guidedState;
	OptixTraversableHandle traversable;
} LaunchParamsPPG;

KRR_NAMESPACE_END