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

	void resize(const Vector2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

	void initialize();

	string getName() const override { return "WavefrontPathTracer"; }

	template <typename F>
	void ParallelFor(int nElements, F&& func) {
		DCHECK_GT(nElements, 0);
		GPUParallelFor(nElements, func);
	}

	void handleHit();
	void handleMiss();
	void generateScatterRays();
	void generateCameraRays(int sampleId);

	KRR_CALLABLE RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	KRR_CALLABLE RayQueue* nextRayQueue(int depth) { return rayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

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
	Vector2i frameSize{ };
	int samplesPerPixel{ 1 };
	int maxDepth{ 10 };
	float probRR{ 0.8 };
	bool enableNEE{ };
	bool debugOutput{ };
	uint debugPixel{ };
	bool enableClamp{ true };
	float clampMax{ 1000 };

	KRR_REGISTER_PASS_DEC(WavefrontPathTracer);
};

KRR_NAMESPACE_END