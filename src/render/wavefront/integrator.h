#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "backend.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

class WavefrontPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<WavefrontPathTracer>;

	WavefrontPathTracer() = default;
	WavefrontPathTracer(Scene& scene);
	~WavefrontPathTracer() = default;

	void resize(const vec2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

	void initialize();

	// cuda utility functions
	template <typename F>
	void Call(F&& func) {
#ifdef KRR_ON_GPU
		GPUParallelFor(1, [=] KRR_DEVICE(int) mutable { func(); });
#else 
		assert(!"should not go here");
#endif
	}

	template <typename F>
	void ParallelFor(int nElements, F&& func) {
#ifdef KRR_ON_GPU
		GPUParallelFor(nElements, func);
#else 
		assert(!"should not go here");
#endif
	}

//private: // extended lambda cannot have private or protected access within its class
	void handleHit(vec4f* frameBuffer);
	void handleMiss(vec4f* frameBuffer);
	void evalDirect(vec4f* frameBuffer);
	void generateCameraRays(int sampleId);

	RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	RayQueue* nextRayQueue(int depth) { return rayQueue[(depth + 1) & 1]; }

	OptiXWavefrontBackend* backend;
	Camera* camera{ };

	// work queues
	RayQueue* rayQueue[2]{ };	// switching bewteen current and next queue
	MissRayQueue* missRayQueue{ };
	HitLightRayQueue* hitLightRayQueue{ };
	ShadowRayQueue* shadowRayQueue{ };

	// path tracing parameters
	int frameId{ 0 };
	int maxQueueSize;
	vec2i frameSize{ };
	int samplesPerPixel{ 1 };
	int maxDepth{ 10 };
};

KRR_NAMESPACE_END