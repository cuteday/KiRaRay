#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "integrator.h"
#include "wavefront.h"
#include "render/profiler/profiler.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN
extern "C" char WAVEFRONT_PTX[];

template <typename... Args>
KRR_DEVICE_FUNCTION void WavefrontPathTracer::debugPrint(uint pixelId, const char *fmt,
														 Args &&...args) {
	if (pixelId == debugPixel) printf(fmt, std::forward<Args>(args)...);
}

void WavefrontPathTracer::initialize() {
	Allocator &alloc = *gpContext->alloc;
	maxQueueSize	 = mFrameSize[0] * mFrameSize[1];
	CUDA_SYNC_CHECK(); // necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) rayQueue[i]->resize(maxQueueSize, alloc);
		else rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}
	if (missRayQueue) missRayQueue->resize(maxQueueSize, alloc);
	else missRayQueue = alloc.new_object<MissRayQueue>(maxQueueSize, alloc);
	if (hitLightRayQueue) hitLightRayQueue->resize(maxQueueSize, alloc);
	else hitLightRayQueue = alloc.new_object<HitLightRayQueue>(maxQueueSize, alloc);
	if (shadowRayQueue) shadowRayQueue->resize(maxQueueSize, alloc);
	else shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
	if (scatterRayQueue) scatterRayQueue->resize(maxQueueSize, alloc);
	else scatterRayQueue = alloc.new_object<ScatterRayQueue>(maxQueueSize, alloc);
	if (pixelState) pixelState->resize(maxQueueSize, alloc);
	else pixelState = alloc.new_object<PixelStateBuffer>(maxQueueSize, alloc);
	cudaDeviceSynchronize();
	if (!camera) camera = alloc.new_object<Camera>();
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParams params = {};
	params.traversable		   = backend->getRootTraversable();
	params.sceneData		   = backend->getSceneData();
	params.currentRayQueue	   = currentRayQueue(depth);
	params.missRayQueue		   = missRayQueue;
	params.hitLightRayQueue	   = hitLightRayQueue;
	params.scatterRayQueue	   = scatterRayQueue;
	params.nextRayQueue		   = nextRayQueue(depth);
	backend->launch(params, "Closest", maxQueueSize, 1, 1);
}

void WavefrontPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParams params = {};
	params.traversable		   = backend->getRootTraversable();
	params.sceneData		   = backend->getSceneData();
	params.shadowRayQueue	   = shadowRayQueue;
	params.pixelState		   = pixelState;
	backend->launch(params, "Shadow", maxQueueSize, 1, 1);
}

void WavefrontPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		hitLightRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(const HitLightWorkItem &w) {
			Color Le		= w.light.L(w.p, w.n, w.uv, w.wo);
			float misWeight = 1;
			// Simple understanding: if the sampled component is a delta func, then
			// it has infinite values and has 1 MIS weights.
			if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
				Light light = w.light;
				Interaction intr(w.p, w.wo, w.n, w.uv);
				float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(light);
				float bsdfPdf  = w.pdf;
				misWeight	   = evalMIS(bsdfPdf, lightPdf);
			}
			pixelState->addRadiance(w.pixelId, Le * w.thp * misWeight);
		});
}

void WavefrontPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData &sceneData = mpScene->mpSceneRT->getSceneData();
	ForAllQueued(
		missRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(const MissRayWorkItem &w) {
			Color L = {};
			Interaction intr(w.ray.origin);
			for (const InfiniteLight &light : *sceneData.infiniteLights) {
				float misWeight = 1;
				if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
					float bsdfPdf  = w.pdf;
					float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
					misWeight	   = evalMIS(bsdfPdf, lightPdf);
				}
				L += light.Li(w.ray.dir) * misWeight;
			}
			pixelState->addRadiance(w.pixelId, w.thp * L);
		});
}

void WavefrontPathTracer::generateScatterRays() {
	PROFILE("Generate scatter rays");
	ForAllQueued(
		scatterRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {
			Sampler sampler = &pixelState->sampler[w.pixelId];
			/*  Russian Roulette: If the path is terminated by this vertex,
				then NEE should not be evaluated */
			if (sampler.get1D() >= probRR)
				return;
			w.thp /= probRR;

			const ShadingData &sd = w.sd;
			Vector3f woLocal	  = sd.frame.toLocal(sd.wo);
			BSDFType bsdfType	  = sd.getBsdfType();
			/* sample direct lighting */
			if (enableNEE && (bsdfType & BSDF_SMOOTH)) {
				SampledLight sampledLight = lightSampler.sample(sampler.get1D());
				Light light				  = sampledLight.light;
				LightSample ls			  = light.sampleLi(sampler.get2D(), { sd.pos, sd.frame.N });
				Ray shadowRay			  = sd.getInteraction().spawnRay(ls.intr);
				Vector3f wiWorld		  = normalize(shadowRay.dir);
				Vector3f wiLocal		  = sd.frame.toLocal(wiWorld);

				float lightPdf	= sampledLight.pdf * ls.pdf;
				float misWeight{1};
				Color bsdfVal	= BxDF::f(sd, woLocal, wiLocal, (int) sd.bsdfType);
				if (enableMIS) {
					float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int) sd.bsdfType);
					misWeight	  = evalMIS(lightPdf, bsdfPdf);
				}
				if (lightPdf > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
					ShadowRayWorkItem sw = {};
					sw.ray				 = shadowRay;
					sw.Li				 = ls.L;
					sw.a	   = w.thp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
					sw.pixelId = w.pixelId;
					sw.tMax	   = 1;
					if (sw.a.any()) shadowRayQueue->push(sw);
				}
			}

			/* sample BSDF */
			BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int) sd.bsdfType);
			if (sample.pdf != 0 && sample.f.any()) {
				Vector3f wiWorld = sd.frame.toWorld(sample.wi);
				RayWorkItem r	 = {};
				Vector3f p		 = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
				r.bsdfType		 = sample.flags;
				r.pdf			 = sample.pdf;
				r.ray			 = { p, wiWorld };
				r.ctx			 = { sd.pos, sd.frame.N };
				r.pixelId		 = w.pixelId;
				r.depth			 = w.depth + 1;
				r.thp			 = w.thp * sample.f * fabs(sample.wi[2]) / sample.pdf;
				if (any(r.thp))
					nextRayQueue(w.depth)->push(r);
			}
		});
}

void WavefrontPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate camera rays");
	RayQueue *cameraRayQueue = currentRayQueue(0);
	ParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			Sampler sampler		= &pixelState->sampler[pixelId];
			Vector2i pixelCoord = { pixelId % mFrameSize[0], pixelId / mFrameSize[0] };
			Ray cameraRay		= camera->getRay(pixelCoord, mFrameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		});
}

void WavefrontPathTracer::resize(const Vector2i &size) {
	RenderPass::resize(size);
	initialize(); // need to resize the queues
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene) {
	initialize();
	mpScene = scene;
	mpScene->initializeSceneRT();
	lightSampler = scene->mpSceneRT->getSceneData().lightSampler;
	if (!backend) {
		backend		= new OptiXBackendImpl();
		auto params = OptiXInitializeParameters()
						  .setPTX(WAVEFRONT_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false);
		backend->initialize(params);
	}
	backend->setScene(*scene);
}

void WavefrontPathTracer::beginFrame() {
	if (!mpScene || !maxQueueSize)
		return;
	PROFILE("Begin frame");
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	ParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) { // reset per-pixel sample state
			Vector2i pixelCoord	   = { pixelId % mFrameSize[0], pixelId / mFrameSize[0] };
			pixelState->L[pixelId] = 0;
			pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameId * samplesPerPixel);
			pixelState->sampler[pixelId].advance(256 * pixelId);
		});
}

void WavefrontPathTracer::render(RenderFrame::SharedPtr frame) {
	if (!mpScene || !maxQueueSize)
		return;
	PROFILE("Wavefront Path Tracer");
	CudaRenderTarget frameBuffer = frame->getCudaRenderTarget();
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
			});
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			traceClosest(depth);
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			if (!depth || !enableNEE || enableMIS) {
				handleHit();
				handleMiss();
			}
			// Break on maximum depth, but incorprate contribution from emissive hits.
			if (depth == maxDepth) break;
			// [STEP#2.3] evaluate materials & bsdfs, and generate shadow rays
			generateScatterRays();
			// [STEP#2.4] trace shadow rays (next event estimation)
			if (enableNEE) traceShadow();
		}
	}
	ParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			Color L = pixelState->L[pixelId] / float(samplesPerPixel);
			if (enableClamp)
				L = clamp(L, 0.f, clampMax);
			frameBuffer.write(Color4f(L, 1), pixelId);
		});
	frameId++;
}

void WavefrontPathTracer::renderUI() {
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &samplesPerPixel);
	ui::InputInt("Max bounces", &maxDepth, 1);
	ui::SliderFloat("Russian roulette", &probRR, 0, 1);
	ui::Checkbox("Enable NEE", &enableNEE);
	// If MIS is disabled while NEE is enabled,
	// The paths that hits the lights will not contribute.
	if (enableNEE) ui::Checkbox("Enable MIS", &enableMIS);
	ui::Text("Debugging");
	ui::Checkbox("Debug output", &debugOutput);
	if (debugOutput)
		ui::InputInt("Debug pixel:", (int *) &debugPixel);
	ui::Checkbox("Clamping pixel value", &enableClamp);
	if (enableClamp)
		ui::DragFloat("Max:", &clampMax, 1, 1, 1e5f, "%.1f");
}

KRR_REGISTER_PASS_DEF(WavefrontPathTracer);
KRR_NAMESPACE_END