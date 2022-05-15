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
	CUDA_SYNC_CHECK();	// necessary
	if (maxQueueSize > 0) {
		ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
			pixelState->sampler[pixelId].initialize(RandomizeStrategy::Owen);
		});
	}
	if (!camera) camera = alloc.new_object<Camera>();
	if (!backend) backend = new OptiXWavefrontBackend();
}

void WavefrontPathTracer::handleHit(){
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		color Le = w.light.L(w.p, w.n, w.uv, w.wo);
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
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		color L = {};
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
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState->sampler[w.pixelId];
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
			// TODO: check why ls.pdf (shape_sample.pdf) can potentially be zero.
			float lightPdf = sampledLight.pdf * ls.pdf;
			float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
			vec3f bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType) * fabs(wiLocal.z);

			float misWeight = evalMIS(lightPdf, bsdfPdf);
			if (!isnan(misWeight) && !isinf(misWeight)) {
				//if(isnan(misWeight))
				//	printf("nee misWeight %f lightPdf %f bsdfPdf %f lightSelect %f lightSample %f\n",
				//		misWeight, lightPdf, bsdfPdf, sampledLight.pdf, ls.pdf);
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
			vec3f wiWorld = sd.fromLocal(sample.wi);
			RayWorkItem r = {};
			vec3f p = offsetRayOrigin(sd.pos, sd.N, wiWorld);
			r.pdf = sample.pdf, 1e-7f;
			r.ray = { p, wiWorld };
			r.ctx = { sd.pos, sd.N };
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
		Sampler sampler = &pixelState->sampler[pixelId];
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
	if (!mpScene || !maxQueueSize) return;
	vec4f* frameBuffer = (vec4f*)frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	// reset per-pixel radiance
		pixelState->L[pixelId] = 0;
	});
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	// reset per-pixel sample state
		vec2i pixelCoord = { pixelId % frameSize.x, pixelId / frameSize.x };
		pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameId * samplesPerPixel);
	});
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::render(CUDABuffer& frame){
	if (!mpScene || !maxQueueSize) return;
	vec4f* frameBuffer = (vec4f*)frame.data();
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
			backend->traceShadow(
				maxQueueSize,
				shadowRayQueue,
				pixelState);
		}
	}
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		vec3f L = vec3f(pixelState->L[pixelId]) / samplesPerPixel; 
		if(enableClamp) L = clamp(L, vec3f(0), vec3f(clampMax));
		frameBuffer[pixelId] = vec4f(L, any(L) ? 1 : 0);
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