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
	void beginFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

	void initialize();

	// cuda utility functions
	template <typename F>
	void Call(F&& func) {
		GPUParallelFor(1, [=] KRR_DEVICE(int) mutable { func(); });
	}

	template <typename F>
	void ParallelFor(int nElements, F&& func) {
		DCHECK_GT(nElements, 0);
		GPUParallelFor(nElements, func);
	}

	// extended lambda cannot have private or protected access within its class
	void handleHit();
	void handleMiss();
	void generateScatterRays();
	void generateCameraRays(int sampleId);

	KRR_CALLABLE RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	KRR_CALLABLE RayQueue* nextRayQueue(int depth) { return rayQueue[(depth & 1) ^ 1]; }

	OptiXWavefrontBackend* backend;
	Camera* camera{ };
	LightSampler lightSampler;

	// work queues
	RayQueue* rayQueue[2]{ };	// switching bewteen current and next queue
	MissRayQueue* missRayQueue{ };
	HitLightRayQueue* hitLightRayQueue{ };
	ShadowRayQueue* shadowRayQueue{ };
	ScatterRayQueue* scatterRayQueue{ };
	PixelStateBuffer* pixelState;

	// custom properties
	bool transparentBackground{ };

	// path tracing parameters
	int frameId{ 0 };
	int maxQueueSize;
	vec2i frameSize{ };
	int samplesPerPixel{ 1 };
	int maxDepth{ 10 };
	float probRR{ 0.8 };
	bool enableNEE{ };
	bool debugOutput{ };
	vec2i debugPixel{ };
	bool enableClamp{ true };
	float clampMax{ 100 };
};

KRR_NAMESPACE_END