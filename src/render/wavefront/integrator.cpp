#include <cuda.h>
#include <cuda_runtime.h>

#include "integrator.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

WavefrontPathTracer::WavefrontPathTracer(Scene& scene){
	setScene(std::shared_ptr<Scene>(&scene));
	initialize();
}

void WavefrontPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	maxQueueSize = frameSize.x * frameSize.y;

	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) alloc.delete_object(rayQueue[i]);
		rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}
	if (missRayQueue) alloc.delete_object(missRayQueue);
	missRayQueue = alloc.new_object<MissRayQueue>(maxQueueSize, alloc);
	if (hitLightRayQueue) alloc.delete_object(hitLightRayQueue);
	hitLightRayQueue = alloc.new_object<HitLightRayQueue>(maxQueueSize, alloc);
	if (shadowRayQueue) alloc.delete_object(shadowRayQueue);
	shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
	
	if (!camera) camera = alloc.new_object<Camera>();
}

void WavefrontPathTracer::handleHit(vec4f* frameBuffer){

}

void WavefrontPathTracer::handleMiss(vec4f* frameBuffer){
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		color Li = {};
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			Li += light.Li(w.ray.dir);
		}
		frameBuffer[w.pixelId] += vec4f(Li, 1.0);
	});
}

void WavefrontPathTracer::generateCameraRays(int sampleId)
{
	RayQueue* cameraRayQueue = currentRayQueue(0);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		LCGSampler sampler;
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		sampler.setPixelSample(pixelCoord, frameId * samplesPerPixel + sampleId);
		Ray cameraRay = {
			camera->getPosition(),
			camera->getRayDir(pixelCoord, frameSize, sampler.get2D())
		};
		cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		//printf("current pixel id: %d, current total camera rays: %d\n", pixelId, cameraRayQueue->size());
	});
}


void WavefrontPathTracer::resize(const vec2i& size){
	frameSize = size;
	initialize();
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene){
	scene->toDevice();
	mpScene = scene;
	backend = new OptiXWavefrontBackend(*scene);
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	// beginFrame()
	if (!mpScene) return;

	*camera = *mpScene->getCamera();

	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		RayQueue* cameraRayQueue = currentRayQueue(0);
#ifdef KRR_ON_GPU
		Call(KRR_DEVICE_LAMBDA() { cameraRayQueue->reset(); });
#endif
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; depth < maxDepth; depth++) {
			RayQueue* nextQueue = nextRayQueue(depth);
			Call(KRR_DEVICE_LAMBDA() {
				nextQueue->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
			});
			// [STEP#2.1] find closest intersections
			backend->traceClosest(
				// TODO whether use exact size of maxQueueSize?
				// Observation: it seems the cuda::automic can not be accessed directly in host code
				maxQueueSize,
				//currentRayQueue(depth)->size(),
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				nextRayQueue(depth));
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit((vec4f*)frame.data());
			handleMiss((vec4f*)frame.data());
			// [STEP#2.3] evaluate materials & bsdfs (nee)

			// [STEP#2.4] trace shadow rays (next event estimation)
			backend->traceShadow(
				maxQueueSize,
				shadowRayQueue);

			// Debug phase
			break;
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