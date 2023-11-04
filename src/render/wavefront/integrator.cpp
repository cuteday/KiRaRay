#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "integrator.h"
#include "wavefront.h"
#include "render/spectrum.h"
#include "render/profiler/profiler.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN
extern "C" char WAVEFRONT_PTX[];

void WavefrontPathTracer::initialize() {
	Allocator &alloc = *gpContext->alloc;
	maxQueueSize	 = getFrameSize()[0] * getFrameSize()[1];
	CUDA_SYNC_CHECK(); // necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) 
		if (rayQueue[i]) rayQueue[i]->resize(maxQueueSize, alloc);
		else rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
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
	if (enableMedium) {
		if (mediumSampleQueue) mediumSampleQueue->resize(maxQueueSize, alloc);
		else mediumSampleQueue = alloc.new_object<MediumSampleQueue>(maxQueueSize, alloc);
		if (mediumScatterQueue) mediumScatterQueue->resize(maxQueueSize, alloc);
		else mediumScatterQueue = alloc.new_object<MediumScatterQueue>(maxQueueSize, alloc);
	}
	cudaDeviceSynchronize();
	if (!camera) camera = alloc.new_object<Camera::CameraData>();
	CUDA_SYNC_CHECK();
}

void WavefrontPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	// Telling whether volume rendering is enabled by mediumSampleQueue is null?
	static LaunchParams params = {};
	params.traversable		   = backend->getRootTraversable();
	params.sceneData		   = backend->getSceneData();
	params.colorSpace		   = KRR_DEFAULT_COLORSPACE;
	params.currentRayQueue	   = currentRayQueue(depth);
	params.missRayQueue		   = missRayQueue;
	params.hitLightRayQueue	   = hitLightRayQueue;
	params.scatterRayQueue	   = scatterRayQueue;
	params.nextRayQueue		   = nextRayQueue(depth);
	params.mediumSampleQueue   = enableMedium ? mediumSampleQueue : nullptr;
	params.pixelState		   = pixelState;
	backend->launch(params, "Closest", maxQueueSize, 1, 1);
}

void WavefrontPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParams params = {};
	params.traversable		   = backend->getRootTraversable();
	params.sceneData		   = backend->getSceneData();
	params.colorSpace		   = KRR_DEFAULT_COLORSPACE;
	params.shadowRayQueue	   = shadowRayQueue;
	params.pixelState		   = pixelState;
	if(enableMedium) backend->launch(params, "ShadowTr", maxQueueSize, 1, 1);
	else backend->launch(params, "Shadow", maxQueueSize, 1, 1);
}

void WavefrontPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(
		hitLightRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(const HitLightWorkItem &w) {
			Spectrum Le = w.light.L(w.p, w.n, w.uv, w.wo, pixelState->lambda[w.pixelId]) * w.thp;
			if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
				Interaction intr(w.p, w.wo, w.n, w.uv);
				float lightPdf = w.light.pdfLi(intr, w.ctx) * lightSampler.pdf(w.light);
				Le /= (w.pl * lightPdf + w.pu).mean();
			} else Le /= w.pu.mean();
			pixelState->addRadiance(w.pixelId, Le);
		});
}

void WavefrontPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData &sceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(
		missRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(const MissRayWorkItem &w) {
			Spectrum L		  = {};
			SampledWavelengths lambda = pixelState->lambda[w.pixelId];
			Interaction intr(w.ray.origin);
			for (const rt::InfiniteLight &light : sceneData.infiniteLights) {
				if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
					float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
					L += light.Li(w.ray.dir, lambda) / (w.pu + w.pl * lightPdf).mean();
				} else L += light.Li(w.ray.dir, lambda) / w.pu.mean();	
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
			if (sampler.get1D() >= probRR) return;
			w.thp /= probRR;
			const SurfaceInteraction &intr = w.intr;
			Vector3f woLocal			   = intr.toLocal(intr.wo);
			BSDFType bsdfType			   = intr.getBsdfType();
			SampledWavelengths lambda	   = pixelState->lambda[w.pixelId];
			/* sample direct lighting */
			if (enableNEE && (bsdfType & BSDF_SMOOTH)) {
				SampledLight sampledLight = lightSampler.sample(sampler.get1D());
				Light light				  = sampledLight.light;
				LightSample ls			  = light.sampleLi(sampler.get2D(), {intr.p, intr.n}, lambda);
				Ray shadowRay			  = intr.spawnRayTo(ls.intr);
				Vector3f wiWorld		  = normalize(shadowRay.dir);
				Vector3f wiLocal		  = intr.toLocal(wiWorld);

				float lightPdf			= sampledLight.pdf * ls.pdf;
				Spectrum bsdfVal = BxDF::f(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
				float bsdfPdf			= light.isDeltaLight()
											  ? 0
											  : BxDF::pdf(intr, woLocal, wiLocal, (int) intr.sd.bsdfType);
				if (lightPdf > 0 && bsdfVal.any()) {
					ShadowRayWorkItem sw = {};
					sw.ray				 = shadowRay;
					sw.Ld				 = ls.L * w.thp * bsdfVal * fabs(wiLocal[2]);
					sw.pu				 = w.pu * bsdfPdf;
					sw.pl				 = w.pu * lightPdf;
					sw.pixelId			 = w.pixelId;
					sw.tMax				 = 1;
					if (sw.Ld.any()) shadowRayQueue->push(sw);
				}
			}

			/* sample BSDF */
			BSDFSample sample = BxDF::sample(intr, woLocal, sampler, (int) intr.sd.bsdfType);
			if (sample.pdf != 0 && sample.f.any()) {
				Vector3f wiWorld = intr.toWorld(sample.wi);
				RayWorkItem r	 = {};
				r.bsdfType		 = sample.flags;
				r.pu			 = w.pu;
				r.pl			 = w.pu / sample.pdf;
				r.ray			 = intr.spawnRayTowards(wiWorld);
				r.ctx			 = { intr.p, intr.n };
				r.pixelId		 = w.pixelId;
				r.depth			 = w.depth + 1;
				r.thp			 = w.thp * sample.f * fabs(sample.wi[2]) / sample.pdf;
				if (any(r.thp)) nextRayQueue(w.depth)->push(r);
			}
		});
}

void WavefrontPathTracer::generateCameraRays(int sampleId) {
	PROFILE("Generate camera rays");
	RayQueue *cameraRayQueue = currentRayQueue(0);
	auto frameSize			 = getFrameSize();
	GPUParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			Sampler sampler		= &pixelState->sampler[pixelId];
			Vector2i pixelCoord = {pixelId % frameSize[0], pixelId / frameSize[0]};
			Ray cameraRay		= camera->getRay(pixelCoord, frameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		}, gpContext->cudaStream);
}

void WavefrontPathTracer::resize(const Vector2i &size) {
	RenderPass::resize(size);
	initialize(); // need to resize the queues
}

void WavefrontPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (!backend) {
		backend		= new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(WAVEFRONT_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRaygenEntry("ShadowTr")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false)
						  .addRayType("ShadowTr", true, true, false);
		backend->initialize(params);
	}
	backend->setScene(scene);
	lightSampler = backend->getSceneData().lightSampler;
	enableMedium = enableMedium && scene->getMedia().size();
	initialize();
}

void WavefrontPathTracer::beginFrame(RenderContext* context) {
	if (!mScene || !maxQueueSize) return;
	PROFILE("Begin frame");
	cudaMemcpyAsync(camera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
			   cudaMemcpyHostToDevice, 0);
	size_t frameIndex = getFrameIndex();
	auto frameSize = getFrameSize();
	GPUParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) { // reset per-pixel sample state
			Vector2i pixelCoord		   = {pixelId % frameSize[0], pixelId / frameSize[0]};
			pixelState->L[pixelId]	   = Spectrum::Zero();
			pixelState->pixel[pixelId] = RGB::Zero();
			pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameIndex * samplesPerPixel);
			pixelState->sampler[pixelId].advance(256 * pixelId);
			pixelState->lambda[pixelId]  = SampledWavelengths::sampleUniform(pixelState->sampler[pixelId].get1D());
	}, gpContext->cudaStream);
}

void WavefrontPathTracer::render(RenderContext *context) {
	if (!mScene || !maxQueueSize) return;
	PROFILE("Wavefront Path Tracer");
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); }, gpContext->cudaStream);
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
				if (enableMedium) {
					mediumSampleQueue->reset();
					mediumScatterQueue->reset();
				}
			}, gpContext->cudaStream);
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			traceClosest(depth);
			// [STEP#2.2] sample medium interaction, and optionally sample in-volume scattering events
			if (enableMedium) {
				sampleMediumInteraction(depth);
				sampleMediumScattering(depth);
			}
			// [STEP#2.3] handle hit and missed rays, contribute to pixels
			handleHit();
			handleMiss();
			// Break on maximum depth, but incorprate contribution from emissive hits.
			if (depth == maxDepth) break;
			// [STEP#2.4] evaluate materials & bsdfs, and generate shadow rays
			generateScatterRays();
			// [STEP#2.5] trace shadow rays (next event estimation)
			if (enableNEE) traceShadow();
		}
		GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			RGB L = pixelState->L[pixelId].toRGB(pixelState->lambda[pixelId],
															*KRR_DEFAULT_COLORSPACE_GPU);
			pixelState->pixel[pixelId] = pixelState->pixel[pixelId] + L;
		}, gpContext->cudaStream);
	}
	GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			RGB pixel = pixelState->pixel[pixelId] / float(samplesPerPixel);
			if (enableClamp) pixel.clamp(0, clampMax);
			frameBuffer.write(RGBA(pixel, 1), pixelId);
		}, gpContext->cudaStream);
}

void WavefrontPathTracer::renderUI() {
	ui::Text("Frame index (seed): %zd", getFrameIndex());
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &samplesPerPixel, 1, 1);
	ui::InputInt("Max bounces", &maxDepth, 1);
	ui::SliderFloat("Russian roulette", &probRR, 0, 1);
	ui::Checkbox("Enable NEE", &enableNEE);
	// If MIS is disabled while NEE is enabled,
	// The paths that hits the lights will not contribute.
	if (mScene->getMedia().size())
		if (ui::Checkbox("Enable medium", &enableMedium)) initialize();
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