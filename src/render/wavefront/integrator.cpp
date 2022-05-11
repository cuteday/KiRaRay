#include <cuda.h>
#include <cuda_runtime.h>

#include "integrator.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

WavefrontPathTracer::WavefrontPathTracer(Scene& scene){
	initialize();
	setScene(std::shared_ptr<Scene>(&scene));
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
	if (scatterRayQueue) alloc.delete_object(scatterRayQueue);
	scatterRayQueue = alloc.new_object<ScatterRayQueue>(maxQueueSize, alloc);
	
	pixelState = SOA<PixelState>(maxQueueSize, alloc);
	
	if (!camera) camera = alloc.new_object<Camera>();
	if (!backend) backend = new OptiXWavefrontBackend();
}

void WavefrontPathTracer::handleHit(vec4f* frameBuffer){
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		color Le = w.light.L(w.p, w.n, w.uv, w.wo);
		frameBuffer[w.pixelId] += vec4f(Le * w.thp, 1);
	});
}

void WavefrontPathTracer::handleMiss(vec4f* frameBuffer){
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		color Li = {};
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			Li += light.Li(w.ray.dir);
		}
		frameBuffer[w.pixelId] += vec4f(w.thp * Li, 1);
	});
}

void WavefrontPathTracer::generateScatterRays(){
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState.sampler[w.pixelId];
		float u = sampler.get1D();
		if (u > probRR) return;
		const ShadingData& sd = w.sd;
		vec3f woLocal = sd.toLocal(sd.wo);
		BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		if (sample.pdf && any(sample.f)) {
			vec3f wiWorld = sd.fromLocal(sample.wi);
			RayWorkItem r = {};
			vec3f p = offsetRayOrigin(sd.pos, sd.N, wiWorld);
			r.ray = { p, wiWorld };
			r.pixelId = w.pixelId;
			r.depth = w.depth + 1;
			r.thp = w.thp * sample.f * fabs(sample.wi.z) / sample.pdf / probRR;
			nextRayQueue(w.depth)->push(r);
		}
	});
}

void WavefrontPathTracer::generateCameraRays(int sampleId){
	RayQueue* cameraRayQueue = currentRayQueue(0);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Sampler sampler = &pixelState.sampler[pixelId];
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		Ray cameraRay = {
			camera->getPosition(),
			camera->getRayDir(pixelCoord, frameSize, sampler.get2D())
		};
		cameraRayQueue->pushCameraRay(cameraRay, pixelId);
	});
}

void WavefrontPathTracer::startPixelSamples(){
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		pixelState.sampler[pixelId].setPixelSample(pixelCoord, frameId);
	});
}

void WavefrontPathTracer::resize(const vec2i& size){
	frameSize = size;
	initialize();		// need to resize the queues
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene){
	scene->toDevice();
	mpScene = scene;
	initialize();
	backend->setScene(*scene);
}

void WavefrontPathTracer::beginFrame(CUDABuffer& frame){
	if (!mpScene) return;
	vec4f* frameBuffer = (vec4f*)frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		frameBuffer[pixelId] = vec4f(vec3f(0), 1);
	});
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	startPixelSamples();
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
#ifdef KRR_ON_GPU
		Call(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
#endif
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; depth < maxDepth; depth++) {
			Call(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
			});
			// [STEP#2.1] find closest intersections
			backend->traceClosest(
				maxQueueSize,	// it seems the cuda::automic can not be accessed directly in host code
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				scatterRayQueue,
				nextRayQueue(depth));
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit((vec4f*)frame.data());
			handleMiss((vec4f*)frame.data());
			// [STEP#2.3] evaluate materials & bsdfs
			generateScatterRays();
			// [STEP#2.4] trace shadow rays (next event estimation)
			backend->traceShadow(
				maxQueueSize,
				shadowRayQueue);
			// Debug phase
			CUDA_SYNC_CHECK();
		}
	}
	CUDA_SYNC_CHECK();
	frameId++;
}

void WavefrontPathTracer::renderUI(){
	if (ui::CollapsingHeader("Wavefront path tracer")) {
		ui::Text("Hello from wavefront path tracer!");
		ui::InputInt("Samples per pixel", &samplesPerPixel);
		ui::SliderInt("Max recursion depth", &maxDepth, 0, 30);
	}
}

KRR_NAMESPACE_END