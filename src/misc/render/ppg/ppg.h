#pragma once
#include "render/wavefront/wavefront.h"
#include "guideditem.h"

NAMESPACE_BEGIN(krr)

class PPGPathTracer;

template <> 
struct LaunchParameters<PPGPathTracer> {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;
	MediumSampleQueue *mediumSampleQueue;

	bool enableTraining;
	PixelStateBuffer* pixelState;
	const RGBColorSpace *colorSpace;
	rt::SceneData sceneData;
	GuidedPathStateBuffer* guidedState;
	OptixTraversableHandle traversable;
};

NAMESPACE_END(krr)