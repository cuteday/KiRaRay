#pragma once
#include "render/wavefront/backend.h"
#include "ppg.h"
#include "guideditem.h"

KRR_NAMESPACE_BEGIN

class PPGPathTracer;

class OptiXPPGBackend : public OptiXWavefrontBackend {
public:
	OptiXPPGBackend() = default;
	OptiXPPGBackend(Scene& scene);
	void setScene(Scene& scene);

	void traceClosest(int numRays,
		RayQueue* currentRayQueue,
		MissRayQueue* missRayQueue,
		HitLightRayQueue* hitLightRayQueue,
		ScatterRayQueue* scatterRayQueue,
		RayQueue* nextRayQueue);

	void traceShadow(int numRays,
		ShadowRayQueue* shadowRayQueue,
		PixelStateBuffer* pixelState,
		GuidedPathStateBuffer* guidedState,
		bool enableTraining);

private:
	friend class PPGPathTracer;
};

KRR_NAMESPACE_END