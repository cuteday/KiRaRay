#include "common.h"
#include "window.h"

#include "util/check.h"
#include "util/film.h"

#include "tree.h"
#include "integrator.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
	static int guiding_trained_frames = 0;
}

void PPGPathTracer::resize(const Vector2i& size) {
	RenderPass::resize(size);
	initialize();		// need to resize the queues
}

void PPGPathTracer::setScene(Scene::SharedPtr scene) {
	scene->toDevice();
	mpScene = scene;
	lightSampler = scene->getSceneData().lightSampler;
	initialize();
	backend->setScene(*scene);
	AABB aabb = scene->getAABB();
	Allocator& alloc = *gpContext->alloc;
	if (m_sdTree) alloc.deallocate_object(m_sdTree);
	m_sdTree = alloc.new_object<STree>(aabb, alloc);
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	WavefrontPathTracer::initialize();
	cudaDeviceSynchronize();
	/* override some default options... */
	enableClamp = false;
	maxDepth = 6;
	probRR = 1;
	Log(Info, "Initializing guided state buffer for SD-Tree...");
	if (guidedPathState) guidedPathState->resize(maxQueueSize, alloc);
	else guidedPathState = alloc.new_object<GuidedPathStateBuffer>(maxQueueSize, alloc);
	if (guidedRayQueue) guidedRayQueue->resize(maxQueueSize, alloc);
	else guidedRayQueue = alloc.new_object<GuidedRayQueue>(maxQueueSize, alloc);
	if (!backend) backend = new OptiXPPGBackend();
	/* @addition VAPG */
	std::cout << "current frame size: " << mFrameSize;
	if (m_image)  m_image->resize(mFrameSize);
	else m_image = alloc.new_object<Film>(mFrameSize);
	if (m_pixelEstimate)  m_pixelEstimate->resize(mFrameSize);
	else m_pixelEstimate = alloc.new_object<Film>(mFrameSize);
	m_image->reset();
	m_pixelEstimate->reset();
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		Color3f Le = w.light.L(w.p, w.n, w.uv, w.wo);
		float misWeight = 1;
		if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
			Light light = w.light;
			Interaction intr(w.p, w.wo, w.n, w.uv);
			float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(light);
			float bsdfPdf = w.pdf;
			misWeight = evalMIS(bsdfPdf, lightPdf);
		}
		Color3f contrib = Le * w.thp * misWeight;
		pixelState->addRadiance(w.pixelId, contrib);
		guidedPathState->recordRadiance(w.pixelId, contrib);
	});
}

void PPGPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem & w) {
		Color3f L = {};
		Interaction intr(w.ray.origin);
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			float misWeight = 1;
			if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
				float bsdfPdf = w.pdf;
				float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
				misWeight = evalMIS(bsdfPdf, lightPdf);
			}
			L += light.Li(w.ray.dir) * misWeight;
		}
		pixelState->addRadiance(w.pixelId, w.thp* L);
		guidedPathState->recordRadiance(w.pixelId, w.thp * L);	// ppg
	});
}

void PPGPathTracer::handleIntersections() {
	PROFILE("Handle intersections");
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState->sampler[w.pixelId];
		if (sampler.get1D() >= probRR) return;
		w.thp /= probRR;
		
		const ShadingData& sd = w.sd;
		BSDFType bsdfType = BxDF::flags(sd, (int)w.sd.bsdfType);
		Vector3f woLocal = sd.frame.toLocal(sd.wo);

		/* Statistics for mixed bsdf-guided sampling */
		float bsdfPdf, dTreePdf;
		DTreeWrapper* dTree = m_sdTree->dTreeWrapper(sd.pos);

		if (enableNEE && (bsdfType & BSDF_SMOOTH)) {
			SampledLight sampledLight = lightSampler.sample(sampler.get1D());
			Light light = sampledLight.light;
			LightSample ls = light.sampleLi(sampler.get2D(), { sd.pos, sd.frame.N });
			Ray shadowRay = sd.getInteraction().spawnRay(ls.intr);
			Vector3f wiWorld = normalize(shadowRay.dir);
			Vector3f wiLocal = sd.frame.toLocal(wiWorld);

			float lightPdf = sampledLight.pdf * ls.pdf;
			float scatterPdf = PPGPathTracer::evalPdf(bsdfPdf, dTreePdf, w.depth, sd, wiLocal,
				m_bsdfSamplingFraction, dTree, bsdfType);
			Color bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType);
			float misWeight = evalMIS(lightPdf, scatterPdf);
			if (misWeight > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
				ShadowRayWorkItem sw = {};
				sw.ray = shadowRay;
				sw.Li = ls.L;
				sw.a = w.thp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
				sw.pixelId = w.pixelId;
				sw.tMax = 1;
				if (sw.a.any()) shadowRayQueue->push(sw);
			}
		}

		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		guidedRayQueue->push(tid); /* unfinished rays */
	});
}

void PPGPathTracer::generateScatterRays() {
	PROFILE("Generate scatter rays");
	ForAllQueued(guidedRayQueue, maxQueueSize, 
		KRR_DEVICE_LAMBDA(const GuidedRayWorkItem &id) {
			const ScatterRayWorkItem w = scatterRayQueue->operator[](id.itemId);
			Sampler sampler			   = &pixelState->sampler[w.pixelId];
			const ShadingData &sd	   = w.sd;
			const BSDFType bsdfType	   = sd.getBsdfType();
			Vector3f woLocal		   = sd.frame.toLocal(sd.wo);

			float bsdfPdf, dTreePdf;
			Vector3f dTreeVoxelSize{};
			DTreeWrapper *dTree = m_sdTree->dTreeWrapper(sd.pos, dTreeVoxelSize);

			BSDFSample sample = PPGPathTracer::sample(sampler, sd, bsdfPdf, dTreePdf, w.depth,
													  m_bsdfSamplingFraction, dTree, bsdfType);
			if (sample.pdf > 0 && sample.f.any()) {
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
				if (any(r.thp)) {
					nextRayQueue(w.depth)->push(r);
					/* guidance... */
					if (r.depth <= MAX_TRAIN_DEPTH) {
						guidedPathState->incrementDepth(r.pixelId, r.ray, dTree, dTreeVoxelSize,
														r.thp, sample.f, r.pdf, bsdfPdf, dTreePdf,
														sample.isSpecular());
					}
				}
			}
	});
}

void PPGPathTracer::render(CUDABuffer& frame) {
	if (!mpScene || !maxQueueSize) return;
	PROFILE("PPG Path Tracer");
	Color4f* frameBuffer = (Color4f*)frame.data();
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
				guidedRayQueue->reset();
			});
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			backend->traceClosest(
				maxQueueSize,	// cuda::atomic can not be accessed directly in host code
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				scatterRayQueue,
				nextRayQueue(depth));
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit();
			if (depth || !transparentBackground) handleMiss();
			// Break on maximum depth, but incorprate contribution from emissive hits.
			if (depth == maxDepth) break;
			// [STEP#2.3] handle intersections and shadow rays
			handleIntersections();
			if (enableNEE)
				backend->traceShadow(maxQueueSize, 
						shadowRayQueue, 
						pixelState, 
						guidedPathState,
						enableLearning);
			// [STEP#2.4] towards next bounce
			generateScatterRays();
		}
	}
	CUDA_SYNC_CHECK();
	frameId++;
}

void PPGPathTracer::beginFrame(CUDABuffer& frame) {
	if (!mpScene || !maxQueueSize) return;
	WavefrontPathTracer::beginFrame(frame);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		guidedPathState->n_vertices[pixelId] = 0;
	});
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::endFrame(CUDABuffer& frame) {
	Color4f* frameBuffer = (Color4f*)frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Color3f L = Color3f(pixelState->L[pixelId]) / samplesPerPixel;
		if (enableClamp) L = clamp(L, 0.f, clampMax);
		frameBuffer[pixelId] = Color4f(L, 1.f);
		if (m_distribution == EDistribution::EFull)
			m_image->put(Color4f(L, 1.f), pixelId);
	});
	if (enableLearning) {
		PROFILE("Training SD-Tree");
		ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
			Sampler sampler = &pixelState->sampler[pixelId];
			Color pixelEstimate = 0.5f;
			if (m_isBuilt && m_distribution == EDistribution::EFull)
				pixelEstimate = m_pixelEstimate->getPixel(pixelId).head<3>();
			guidedPathState->commitAll(pixelId, m_sdTree, 1.f, m_spatialFilter, m_directionalFilter, m_bsdfSamplingFractionLoss, sampler,
										   m_distribution, pixelEstimate);
		});
		++guiding_trained_frames;
	}
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::renderUI() {
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &samplesPerPixel);
	ui::InputInt("Max bounces", &maxDepth, 1);
	ui::SliderFloat("Russian roulette", &probRR, 0, 1);
	ui::Checkbox("Enable NEE", &enableNEE);

	ui::Text("Path guiding");
	const static char *distributionNames[] = { "Radiance", "Partial", "Full" };
	ui::Text("Target distribution mode: %s", distributionNames[(int)m_distribution]);
	ui::Checkbox("Enable learning", &enableLearning);
	ui::Checkbox("Enable guiding", &enableGuiding);
	ui::Text("Current iteration: %d", m_iter);
	ui::DragFloat("Bsdf sampling fraction", &m_bsdfSamplingFraction, 0.01, 0, 1);
	int train_frames_this_iteration = (1 << m_iter) * m_sppPerPass ;
	ui::Text("Frames this iteration: %d / %d", 
		guiding_trained_frames, train_frames_this_iteration);
	ui::ProgressBar((float)guiding_trained_frames / train_frames_this_iteration);
	if (ui::Button("Next guiding iteration")) {
		buildSDTree();		// this is performed at the end of each iteration
		guiding_trained_frames = 0;
		m_sdTree->gatherStatistics();
		resetSDTree();		// this is performed at the beginning of each iteration
		*m_pixelEstimate = *m_image;
		m_image->reset();
		CUDA_SYNC_CHECK();
		m_iter++;
	}
	if (ui::CollapsingHeader("Advanced guiding options")) {
		if (ui::Button("Reset guiding")) {
			cudaDeviceSynchronize();
			m_sdTree->clear();
			m_isBuilt = false;
			m_iter = guiding_trained_frames = 0;
			CUDA_SYNC_CHECK();
		}
		ui::DragInt("Spp per pass", &m_sppPerPass, 1, 1, 100);
		ui::DragInt("S-tree threshold", &m_sTreeThreshold, 1, 1000, 100000, "%d");
		ui::DragFloat("D-tree threshold", &m_dTreeThreshold, 1e-3, 1e-3, 0.1, "%.3f");
	}
	ui::Text("Debugging");
	ui::Checkbox("Debug output", &debugOutput);
	if (debugOutput) {
		ui::InputInt2("Debug pixel:", (int*)&debugPixel);
	}
	ui::Checkbox("Clamping pixel value", &enableClamp);
	if (enableClamp) 
		ui::DragFloat("Max:", &clampMax, 1, 500);
	if (ui::CollapsingHeader("Misc")) {
		ui::Checkbox("Transparent background", &transparentBackground);
	}
}


KRR_CALLABLE BSDFSample PPGPathTracer::sample(Sampler& sampler, 
	const ShadingData& sd, float& bsdfPdf, float& dTreePdf, int depth,
	float bsdfSamplingFraction, const DTreeWrapper* dTree, BSDFType bsdfType) const {
	BSDFSample sample = {};
	Vector3f woLocal = sd.frame.toLocal(sd.wo);

	if (!m_isBuilt || !dTree || !enableGuiding || !(bsdfType & BSDF_SMOOTH) 
		|| bsdfSamplingFraction == 1 || depth >= MAX_GUIDED_DEPTH) {
		sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		bsdfPdf = sample.pdf;
		dTreePdf = 0;
		return sample;
	}

	Vector3f result;
	if (bsdfSamplingFraction > 0 && sampler.get1D() < bsdfSamplingFraction) {
		sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		bsdfPdf = sample.pdf;
		dTreePdf = dTree->pdf(sd.frame.toWorld(sample.wi));
		sample.pdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
		return sample;
	}
	else {
		sample.wi = sd.frame.toLocal(dTree->sample(sampler));
		sample.f = BxDF::f(sd, woLocal, sample.wi, (int)sd.bsdfType);
		sample.flags = BSDF_GLOSSY | (SameHemisphere(sample.wi, woLocal) ?
			BSDF_REFLECTION : BSDF_TRANSMISSION);
		sample.pdf = evalPdf(bsdfPdf, dTreePdf, depth, sd, sample.wi,
			bsdfSamplingFraction, dTree, sample.flags /*The bsdf lobe type is needed (in case for delta lobes)*/);
		return sample;
	}
}

KRR_CALLABLE float PPGPathTracer::evalPdf(float& bsdfPdf, float& dTreePdf, int depth,
	const ShadingData& sd, Vector3f wiLocal, float alpha, const DTreeWrapper* dTree, BSDFType bsdfType) const {
	Vector3f woLocal = sd.frame.toLocal(sd.wo);
	
	bsdfPdf = dTreePdf = 0;
	if (!m_isBuilt || !dTree || !enableGuiding || !(bsdfType & BSDF_SMOOTH) 
		|| alpha == 1 || depth >= MAX_GUIDED_DEPTH) {
		return bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
	}
	if (alpha > 0) {
		bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
		if (isinf(bsdfPdf) || isnan(bsdfPdf)) {
			return bsdfPdf = dTreePdf = 0;
		}
	}
	if (alpha < 1) {
		dTreePdf = dTree->pdf(sd.frame.toWorld(wiLocal));
	}
	return alpha * bsdfPdf + (1 - alpha) * dTreePdf;
}

KRR_REGISTER_PASS_DEF(PPGPathTracer);
KRR_NAMESPACE_END