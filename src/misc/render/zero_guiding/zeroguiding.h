#pragma once
#include "render/wavefront/wavefront.h"
#include "guideditem.h"

KRR_NAMESPACE_BEGIN

class ZeroGuidingPT;

template <> 
struct LaunchParameters<ZeroGuidingPT> {
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

KRR_NAMESPACE_END