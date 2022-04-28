#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"
#include "backend.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

class WavefrontPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<WavefrontPathTracer>;

	WavefrontPathTracer() = default;
	WavefrontPathTracer(Scene& scene);
	
	void resize(const vec2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

protected:
	void initialize();

	void handleHit();
	void handleMiss();
	void evalDirect();

private:
	// utility functions
	RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	RayQueue* nextRayQueue(int depth) { return rayQueue[(depth + 1) & 1]; }
	
	void generateCameraRays(int sampleId);

	OptiXWavefrontBackend backend;

	Scene::SharedPtr scene;
	Camera* camera;


	// work queues
	RayQueue* rayQueue[2];	// switching bewteen current and next queue

	// path tracing parameters
	int frameId{ 0 };
	int maxQueueSize;
	vec2i frameSize{ };
	int samplesPerPixel{ 1 };
	int maxDepth{ 10 };
};

KRR_NAMESPACE_END