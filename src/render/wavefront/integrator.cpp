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
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		pixelState.sampler[pixelId].initialize();
	});
	if (!camera) camera = alloc.new_object<Camera>();
	if (!backend) backend = new OptiXWavefrontBackend();
}

void WavefrontPathTracer::handleHit(){
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		color Le = w.light.L(w.p, w.n, w.uv, w.wo);
		color L = vec3f(pixelState.L[w.pixelId]) + Le * w.thp;
		pixelState.L[w.pixelId] = L;
	});
}

void WavefrontPathTracer::handleMiss(){
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		color Li = {};
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			Li += light.Li(w.ray.dir);
		}
		color L = vec3f(pixelState.L[w.pixelId]) + Li * w.thp;
		pixelState.L[w.pixelId] = L;
	});
}

void WavefrontPathTracer::generateScatterRays(){
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState.sampler[w.pixelId];
		const ShadingData& sd = w.sd;
		vec3f woLocal = sd.toLocal(sd.wo);

		/* sample direct lighting */
		if (enableNEE) {
			float u = sampler.get1D();
			SampledLight sampledLight = lightSampler.sample(u);
			Light light = sampledLight.light;
			LightSample ls = light.sampleLi(sampler.get2D(), { sd.pos, sd.N });
			vec3f wiWorld = normalize(ls.intr.p - sd.pos);
			vec3f wiLocal = sd.toLocal(wiWorld);

			float lightPdf = sampledLight.pdf * ls.pdf;
			float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
			vec3f bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal.z);

			float misWeight = evalMIS(lightPdf, bsdfPdf);
			Ray shadowRay = sd.getInteraction().spawnRay(ls.intr);

		}

		/* sample BSDF */
		float u = sampler.get1D();
		if (u > probRR) return;		// Russian Roulette
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

void WavefrontPathTracer::resize(const vec2i& size){
	frameSize = size;
	initialize();		// need to resize the queues
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene){
	scene->toDevice();
	mpScene = scene;
	lightSampler = scene->getSceneData().lightSampler;
	initialize();
	backend->setScene(*scene);
}

void WavefrontPathTracer::beginFrame(CUDABuffer& frame){
	if (!mpScene) return;
	vec4f* frameBuffer = (vec4f*)frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	// reset per-pixel radiance
		pixelState.L[pixelId] = 0;
	});
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	// reset per-pixel sample state
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		pixelState.sampler[pixelId].setPixelSample(pixelCoord, frameId);
	});
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	vec4f* frameBuffer = (vec4f*)frame.data();
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
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			backend->traceClosest(
				maxQueueSize,	// cuda::automic can not be accessed directly in host code
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				scatterRayQueue,
				nextRayQueue(depth));
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit();
			handleMiss();
			// [STEP#2.3] evaluate materials & bsdfs
			generateScatterRays();
			// [STEP#2.4] trace shadow rays (next event estimation)
			backend->traceShadow(
				maxQueueSize,
				shadowRayQueue,
				&pixelState);
		}
	}
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		frameBuffer[pixelId] = vec4f(vec3f(pixelState.L[pixelId]) / samplesPerPixel, 1);
	});
	CUDA_SYNC_CHECK();
	frameId++;
}

void WavefrontPathTracer::renderUI(){
	if (ui::CollapsingHeader("Wavefront path tracer")) {
		ui::Text("Hello from wavefront path tracer!");
		ui::InputInt("Samples per pixel", &samplesPerPixel);
		ui::SliderInt("Max recursion depth", &maxDepth, 0, 30);
		ui::Checkbox("Enable NEE", &enableNEE);
	}
}

KRR_NAMESPACE_END