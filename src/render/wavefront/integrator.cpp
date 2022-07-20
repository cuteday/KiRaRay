#include <cuda.h>
#include <cuda_runtime.h>

#include "integrator.h"
#include "workqueue.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

WavefrontPathTracer::WavefrontPathTracer(Scene& scene){
	initialize();
	setScene(std::shared_ptr<Scene>(&scene));
}

void WavefrontPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	maxQueueSize = frameSize[0] * frameSize[1];
	CUDA_SYNC_CHECK();	// necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) rayQueue[i]->resize(maxQueueSize, alloc);
		else rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}
	if (missRayQueue)  missRayQueue->resize(maxQueueSize, alloc);
	else missRayQueue = alloc.new_object<MissRayQueue>(maxQueueSize, alloc);
	if (hitLightRayQueue)  hitLightRayQueue->resize(maxQueueSize, alloc);
	else hitLightRayQueue = alloc.new_object<HitLightRayQueue>(maxQueueSize, alloc);
	if (shadowRayQueue) shadowRayQueue->resize(maxQueueSize, alloc);
	else shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
	if (scatterRayQueue) scatterRayQueue->resize(maxQueueSize, alloc);
	else scatterRayQueue = alloc.new_object<ScatterRayQueue>(maxQueueSize, alloc);
	if (pixelState) pixelState->resize(maxQueueSize, alloc);
	else pixelState = alloc.new_object<PixelStateBuffer>(maxQueueSize, alloc);
	cudaDeviceSynchronize();	
	if (!camera) camera = alloc.new_object<Camera>();
	if (!backend) backend = new OptiXWavefrontBackend();
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::handleHit(){
	PROFILE("Process intersected rays");
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		Color Le = w.light.L(w.p, w.n, w.uv, w.wo);
		float misWeight = 1;
		if (enableNEE && w.depth) {
			Light light = w.light;
			Interaction intr(w.p, w.wo, w.n, w.uv);
			float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(light);
			float bsdfPdf = w.pdf;
			misWeight = evalMIS(bsdfPdf, lightPdf);
		}
		pixelState->addRadiance(w.pixelId, Le * w.thp * misWeight);
	});
}

void WavefrontPathTracer::handleMiss(){
	PROFILE("Process escaped rays");
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		Color L = {};
		Interaction intr(w.ray.origin);
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			float misWeight = 1;
			if (enableNEE && w.depth) {
				float bsdfPdf = w.pdf;
				float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
				misWeight = evalMIS(bsdfPdf, lightPdf);
			}
			L += light.Li(w.ray.dir) * misWeight;
		}
		pixelState->addRadiance(w.pixelId, w.thp * L);
	});
}

void WavefrontPathTracer::generateScatterRays(){
	PROFILE("Generate scatter rays");
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState->sampler[w.pixelId];
		const ShadingData& sd = w.sd;
		Vec3f woLocal = sd.frame.toLocal(sd.wo);

		/* sample direct lighting */
		if (enableNEE) {
			float u = sampler.get1D();
			SampledLight sampledLight = lightSampler.sample(u);
			Light light = sampledLight.light;
			LightSample ls = light.sampleLi(sampler.get2D(), { sd.pos, sd.frame.N });
			Vec3f wiWorld = normalize(ls.intr.p - sd.pos);
			Vec3f wiLocal = sd.frame.toLocal(wiWorld);
			
			float lightPdf = sampledLight.pdf * ls.pdf;
			float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
			Color bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal[2]);
			float misWeight = evalMIS(lightPdf, bsdfPdf);
			// TODO: check why ls.pdf (shape_sample.pdf) can potentially be zero.
			//if (isnan(misWeight))
			//	printf("nee misWeight %f lightPdf %f bsdfPdf %f lightSelect %f lightSample %f\n",
			//		misWeight, lightPdf, bsdfPdf, sampledLight.pdf, ls.pdf);
			if (!isnan(misWeight) && !isinf(misWeight)) {
				Ray shadowRay = sd.getInteraction().spawnRay(ls.intr);
				ShadowRayWorkItem sw = {};
				sw.ray = shadowRay;
				sw.Li = ls.L;
				sw.a = w.thp * misWeight * bsdfVal / lightPdf;
				sw.pixelId = w.pixelId;
				sw.tMax = 1;
				shadowRayQueue->push(sw);
			}
		}

		/* sample BSDF */
		float u = sampler.get1D();
		if (u > probRR) return;		// Russian Roulette
		BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		if (sample.pdf && any(sample.f)) {
			Vec3f wiWorld = sd.frame.toWorld(sample.wi);
			RayWorkItem r = {};
			Vec3f p = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
			r.pdf = sample.pdf, 1e-7f;
			r.ray = { p, wiWorld };
			r.ctx = { sd.pos, sd.frame.N };
			r.pixelId = w.pixelId;
			r.depth = w.depth + 1;
			r.thp = w.thp * sample.f * fabs(sample.wi[2]) / sample.pdf / probRR;
			nextRayQueue(w.depth)->push(r);
		}
	});
}

void WavefrontPathTracer::generateCameraRays(int sampleId){
	PROFILE("Generate camera rays");
	RayQueue* cameraRayQueue = currentRayQueue(0);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Sampler sampler = &pixelState->sampler[pixelId];
		Vec2i pixelCoord = { pixelId % frameSize[0], pixelId / frameSize[0] };
		Ray cameraRay = camera->getRay(pixelCoord, frameSize, sampler);
		cameraRayQueue->pushCameraRay(cameraRay, pixelId);
	});
}

void WavefrontPathTracer::resize(const Vec2i& size){
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
	if (!mpScene || !maxQueueSize) return;
	PROFILE("Begin frame");
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	// reset per-pixel sample state
		Vec2i pixelCoord = { pixelId % frameSize[0], pixelId / frameSize[0] };
		pixelState->L[pixelId] = 0;
		pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameId * samplesPerPixel);
		pixelState->sampler[pixelId].advance(256 * pixelId);
	});
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	if (!mpScene || !maxQueueSize) return;
	PROFILE("Wavefront Path Tracer");
	Color4f *frameBuffer = (Color4f *) frame.data();
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		Call(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
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
			if (depth || !transparentBackground) handleMiss();
			// [STEP#2.3] evaluate materials & bsdfs
			generateScatterRays();
			// [STEP#2.4] trace shadow rays (next event estimation)
			if (enableNEE)
				backend->traceShadow(
					maxQueueSize,
					shadowRayQueue,
					pixelState);
		}
	}
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Color L = pixelState->L[pixelId] / float(samplesPerPixel);
		if (enableClamp) L = clamp(L, 0.f, clampMax);
		frameBuffer[pixelId] = Color4f(L, 1);
	});
	frameId++;
}

void WavefrontPathTracer::renderUI(){
	if (ui::CollapsingHeader("Wavefront path tracer")) {
		ui::InputInt("Samples per pixel", &samplesPerPixel);
		ui::SliderInt("Max recursion depth", &maxDepth, 0, 30);
		ui::Checkbox("Enable NEE", &enableNEE);
		ui::Text("Debugging");
		ui::Checkbox("Debug output", &debugOutput);
		if (debugOutput) {
			ui::SameLine();
			ui::InputInt2("Debug pixel:", (int*)&debugPixel);
		}
		ui::Checkbox("Clamping pixel value", &enableClamp);
		if (enableClamp) {
			ui::SameLine();
			ui::DragFloat("Max:", &clampMax, 1, 1, 500);
		}
		ui::Text("Misc");
		ui::Checkbox("Transparent background", &transparentBackground);
	}
}

KRR_NAMESPACE_END