#include "common.h"
#include "window.h"
#include "file.h"

#include "util/check.h"
#include "util/film.h"

#include "tree.h"
#include "integrator.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
static size_t guiding_trained_frames		  = 0;
static size_t train_frames_this_iteration	  = 0;
const static char *spatial_filter_names[]	  = { "Nearest", "StochasticBox", "Box" };
const static char *directional_filter_names[] = { "Nearest", "Box" };
const static char *distribution_names[]		  = { "Radiance", "Partial", "Full" };
}

void PPGPathTracer::resize(const Vector2i& size) {
	RenderPass::resize(size);
	initialize();		// need to resize the queues
}

void PPGPathTracer::setScene(Scene::SharedPtr scene) {
	scene->initializeSceneRT();
	mpScene = scene;
	lightSampler = scene->mpSceneRT->getSceneData().lightSampler;
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
	if (guidedPathState) guidedPathState->resize(maxQueueSize, alloc);
	else guidedPathState = alloc.new_object<GuidedPathStateBuffer>(maxQueueSize, alloc);
	if (guidedRayQueue) guidedRayQueue->resize(maxQueueSize, alloc);
	else guidedRayQueue = alloc.new_object<GuidedRayQueue>(maxQueueSize, alloc);
	if (!backend) backend = new OptiXPPGBackend();
	/* @addition VAPG */
	if (m_image)  m_image->resize(mFrameSize);
	else m_image = alloc.new_object<Film>(mFrameSize);
	if (m_pixelEstimate)  m_pixelEstimate->resize(mFrameSize);
	else m_pixelEstimate = alloc.new_object<Film>(mFrameSize);
	m_image->reset();
	m_pixelEstimate->reset();
	m_task.reset();
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
	const rt::SceneData& sceneData = mpScene->mpSceneRT->getSceneData();
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
		BSDFType bsdfType = sd.getBsdfType();
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

void PPGPathTracer::render(RenderFrame::SharedPtr frame) {
	if (!mpScene || !maxQueueSize) return;
	PROFILE("PPG Path Tracer");
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
	//CUDA_SYNC_CHECK();
	// write results of the current frame...
	CudaRenderTarget frameBuffer = frame->getCudaRenderTarget();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
		Color3f L = Color3f(pixelState->L[pixelId]) / samplesPerPixel;
		if (enableClamp) L = clamp(L, 0.f, clampMax);
		m_image->put(Color4f(L, 1.f), pixelId);
		if (m_renderMode == RenderMode::Interactive)
			frameBuffer.write(Color4f(L, 1), pixelId);
		else if (m_renderMode == RenderMode::Offline)
			frameBuffer.write(m_image->getPixel(pixelId), pixelId);
	});
}

void PPGPathTracer::beginFrame() {
	if (!mpScene || !maxQueueSize) return;
	WavefrontPathTracer::beginFrame();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		guidedPathState->n_vertices[pixelId] = 0;
	});
	// [offline mode] always training when auto-train enabled
	// but the last iteration (the render iteration) do not need training anymore.
	enableLearning = (enableLearning || m_autoBuild) && !m_isFinalIter;	
	train_frames_this_iteration = (1 << m_iter) * m_sppPerPass;
	//CUDA_SYNC_CHECK();
}

void PPGPathTracer::endFrame() {
	frameId++;
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
	m_task.tickFrame();
	if (m_task.isFinished() || (m_isFinalIter &&
		guiding_trained_frames >= train_frames_this_iteration)) {
		gpContext->requestExit();
	}
	if (m_autoBuild && !m_isFinalIter && 
		guiding_trained_frames >= train_frames_this_iteration) {
		nextIteration();
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
	ui::Text("Target distribution mode: %s", distribution_names[(int)m_distribution]);
	ui::Combo("Spatial filter", (int*) & m_spatialFilter, spatial_filter_names, 3);
	ui::Combo("Directional filter", (int *) &m_directionalFilter, directional_filter_names, 2);
	ui::Checkbox("Auto rebuild", &m_autoBuild);
	if (!m_autoBuild) ui::Checkbox("Enable learning", &enableLearning);
	ui::Checkbox("Enable guiding", &enableGuiding);
	ui::Text("Current iteration: %d", m_iter);
	ui::DragFloat("Bsdf sampling fraction", &m_bsdfSamplingFraction, 0.01, 0, 1);
	ui::Text("Frames this iteration: %d / %d", 
		guiding_trained_frames, train_frames_this_iteration);
	ui::ProgressBar((float)guiding_trained_frames / train_frames_this_iteration);
	if (ui::Button("Next guiding iteration")) {
		nextIteration();
	}
	if (ui::CollapsingHeader("Task progress")) {
		m_task.renderUI();
	}
	if (ui::CollapsingHeader("Advanced guiding options")) {
		if (ui::Button("Reset guiding")) {
			resetGuiding();
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

void PPGPathTracer::resetGuiding() {
	cudaDeviceSynchronize();
	m_sdTree->clear();
	m_image->reset();
	m_isBuilt = m_isFinalIter = false;
	m_iter = guiding_trained_frames = 0;
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::nextIteration() {
	if (m_isFinalIter) {
		Log(Warning, "Attempting to rebuild SD-Tree in the last iteration");
		return;
	}
	buildSDTree();		// this is performed at the end of each iteration
	guiding_trained_frames = 0;
	m_sdTree->gatherStatistics();
	resetSDTree();		// this is performed at the beginning of each iteration
	if (m_distribution == EDistribution::EFull) {
		*m_pixelEstimate = *m_image;
		filterFrame(m_pixelEstimate);
		if (m_saveIntermediate)
			m_pixelEstimate->save(File::outputDir() /
								  ("iteration_" + std::to_string(m_iter) + ".exr"));
	}

	if (m_sampleCombination == ESampleCombination::EDiscardWithAutomaticBudget)
		m_image->reset();	// discard previous samples each iteration
	CUDA_SYNC_CHECK();

	m_iter++;
	/* Determine whether the current iteration is the final iteration. */
	if (m_trainingIterations > 0 && m_iter == m_trainingIterations)
		m_isFinalIter = true;
	size_t train_frames_next_iter = (1 << m_iter) * m_sppPerPass;
	Budget budget = m_task.getBudget();
	if (budget.type == BudgetType::Time) {
		if (m_task.getProgress() > 0.33f)
			m_isFinalIter = true;
	} else if (budget.type == BudgetType::Spp) {
		size_t remaining_spp = budget.value * (1.f - m_task.getProgress());
		if (remaining_spp - train_frames_next_iter < 2 * train_frames_next_iter)
			// Final iteration must use at least half of the SPP budget.
			m_isFinalIter = true;
	}
}

void PPGPathTracer::finalize() { 
	cudaDeviceSynchronize();
	string output_name = gpContext->getGlobalConfig().contains("name") ? 
		gpContext->getGlobalConfig()["name"] : "result";
	fs::path save_path = File::outputDir() / (output_name + ".exr");
	m_image->save(save_path);
	Log(Info, "Total SPP: %zd, elapsed time: %.1f", 
		m_task.getCurrentSpp(), m_task.getElapsedTime());
	Log(Success, "Task finished, saving results to %s", save_path.string().c_str());
	CUDA_SYNC_CHECK();
}

KRR_CALLABLE BSDFSample PPGPathTracer::sample(Sampler& sampler, 
	const ShadingData& sd, float& bsdfPdf, float& dTreePdf, int depth,
	float bsdfSamplingFraction, const DTreeWrapper* dTree, BSDFType bsdfType) const {
	BSDFSample sample = {};
	Vector3f woLocal = sd.frame.toLocal(sd.wo);

	if (!m_isBuilt || !dTree || !enableGuiding || (bsdfType & BSDF_SPECULAR)
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
	if (!m_isBuilt || !dTree || !enableGuiding || (bsdfType & BSDF_SPECULAR)
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

void PPGPathTracer::filterFrame(Film *image) { 
	/* PixelData format: RGBAlphaWeight */
	using PixelData = Film::WeightedPixel;
	Vector2i size = image->size();
	size_t n_pixels = size[0] * size[1];
	std::vector<PixelData> pixels(n_pixels);
	float *data = reinterpret_cast<float*>(pixels.data());
	
	int blackPixels = 0;
	const int fpp = sizeof(PixelData) / sizeof(float);
	image->getInternalBuffer().copy_to_host(reinterpret_cast<PixelData *>(data), n_pixels);

	constexpr int FILTER_ITERATIONS = 3;
	constexpr int SPECTRUM_SAMPLES	= Color4f::dim;

	for (int i = 0, j = size[0] * size[1]; i < j; ++i) {
		int isBlack = true;
		for (int chan = 0; chan < SPECTRUM_SAMPLES; ++chan)
			isBlack &= data[chan + i * fpp] == 0.f;
		blackPixels += isBlack;
	}

	float blackDensity = 1.f / (sqrtf(1.f - blackPixels / float(size[0] * size[1])) + 1e-3);
	const int FILTER_WIDTH = max(min(int(3 * blackDensity), min(size[0], size[1]) - 1), 1);

	float *stack   = new float[FILTER_WIDTH];
	auto boxFilter = [=](float *data, int stride, int n) {
		assert(n > FILTER_WIDTH);

		double avg = FILTER_WIDTH * data[0];
		for (int i = 0; i < FILTER_WIDTH; ++i) {
			stack[i] = data[0];
			avg += data[i * stride];
		}

		for (int i = 0; i < n; ++i) {
			avg += data[std::min(i + FILTER_WIDTH, n - 1) * stride];
			float newVal = avg;
			avg -= stack[i % FILTER_WIDTH];

			stack[i % FILTER_WIDTH] = data[i * stride];
			data[i * stride]		= newVal;
		}
	};

	for (int i = 0, j = size[0] * size[1]; i < j; ++i) {
		float &weight = data[SPECTRUM_SAMPLES + 1 + i * fpp];
		if (weight > 0.f)
			for (int chan = 0; chan < SPECTRUM_SAMPLES; ++chan)
				data[chan + i * fpp] /= weight;
		weight = 1.f;
	}

	for (int chan = 0; chan < SPECTRUM_SAMPLES; ++chan) {
		for (int iter = 0; iter < FILTER_ITERATIONS; ++iter) {
			for (int x = 0; x < size[0]; ++x)
				boxFilter(data + chan + x * fpp, size[0] * fpp, size[1]);
			for (int y = 0; y < size[1]; ++y)
				boxFilter(data + chan + y * size[0] * fpp, fpp, size[0]);
		}
		for (int i = 0, j = size[0] * size[1]; i < j; ++i) {
			float norm = 1.f / std::pow(2 * FILTER_WIDTH + 1, 2 * FILTER_ITERATIONS);
			data[chan + i * fpp] *= norm;
			// apply offset to avoid numerical instabilities
			data[chan + i * fpp] += 1e-3;
		}
	}
	delete[] stack;

	image->getInternalBuffer().copy_from_host(reinterpret_cast<PixelData *>(data), n_pixels);
}

KRR_REGISTER_PASS_DEF(PPGPathTracer);
KRR_NAMESPACE_END