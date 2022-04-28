#include "integrator.h"

#include "device/cuda.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

WavefrontPathTracer::WavefrontPathTracer(Scene& scene){
	setScene(std::shared_ptr<Scene>(&scene));
	initialize();
}

void WavefrontPathTracer::initialize(){
	Allocator alloc = Allocator(&CUDATrackedMemory::singleton);
	maxQueueSize = frameSize.x * frameSize.y;

	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) alloc.delete_object(rayQueue[i]);
		rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}

	if (!camera) alloc.new_object<Camera>();
}

void WavefrontPathTracer::generateCameraRays(int sampleId)
{
	int nCameraRays = frameSize.x * frameSize.y;
	RayQueue* cameraRayQueue = currentRayQueue(0);
	GPUParrallelFor(nCameraRays, KRR_DEVICE_LAMBDA(int pixelId){
		LCGSampler sampler;
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		sampler.setPixelSample(pixelCoord, frameId * samplesPerPixel + sampleId);
		Ray cameraRay = {
			camera->getPosition(),
			camera->getRayDir(pixelCoord, frameSize, sampler.get2D())
		};
		cameraRayQueue->pushCameraRay(cameraRay, pixelId);
	});
}

void WavefrontPathTracer::resize(const vec2i& size){
	initialize();
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene){
	backend = OptiXWavefrontBackend(*scene);
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	// beginFrame()
	*camera = *scene->getCamera();

	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		RayQueue* cameraRayQueue = currentRayQueue(0);
		GPUCall(KRR_DEVICE_LAMBDA() { cameraRayQueue->reset(); });
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; depth < maxDepth; depth++) {
			// [STEP#2.1] find closest intersections

			// [STEP#2.2] handle hit and missed rays, contribute to pixels

			// [STEP#2.3] evaluate materials & bsdfs

			// [STEP#2.4] trace shadow rays (next event estimation)

		}
	}
	CUDA_SYNC_CHECK();
	frameId++;
}

void WavefrontPathTracer::renderUI()
{
	if (ui::CollapsingHeader("Wavefront path tracer")) {
		ui::Text("Hello from wavefront path tracer!");
		ui::InputInt("Samples per pixel", &samplesPerPixel);
		ui::SliderInt("Max recursion depth", &maxDepth, 0, 30);
	}
}

KRR_NAMESPACE_END