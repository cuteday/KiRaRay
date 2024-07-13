#include "common.h"
#include "window.h"
#include "file.h"

#include "util/check.h"
#include "util/film.h"

#include "ppg.h"
#include "tree.h"
#include "integrator.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)
extern "C" char PPG_PTX[];

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
	mScene = scene;
	if (!backend) backend		= new OptixBackend();
	backend->setScene(scene);
	auto params = OptixInitializeParameters()
						.setPTX(PPG_PTX)
						.addRaygenEntry("Closest")
						.addRaygenEntry("Shadow")
						.addRaygenEntry("ShadowTr")
						.addRayType("Closest", true, true, false)
						.addRayType("Shadow", false, true, false)
						.addRayType("ShadowTr", true, true, false)
						.setMaxTraversableDepth(scene->getMaxGraphDepth());
	backend->initialize(params);
	backend->buildShaderBindingTable();
	lightSampler	 = backend->getSceneData().lightSampler;
	AABB aabb = scene->getBoundingBox();
	Allocator& alloc = *gpContext->alloc;
	if (m_sdTree) alloc.deallocate_object(m_sdTree);
	m_sdTree = alloc.new_object<STree>(aabb, alloc);
	initialize();
}

void PPGPathTracer::initialize() {
	Allocator& alloc = *gpContext->alloc;
	WavefrontPathTracer::initialize();
	// We need this since CUDA 12 seems to have reduced the default stack size,
	// However, SD-Tree has some recursive routines that may exceed that size;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
	cudaDeviceSynchronize();
	if (guidedPathState) guidedPathState->resize(maxQueueSize, alloc);
	else guidedPathState = alloc.new_object<GuidedPathStateBuffer>(maxQueueSize, alloc);
	if (guidedRayQueue) guidedRayQueue->resize(maxQueueSize, alloc);
	else guidedRayQueue = alloc.new_object<GuidedRayQueue>(maxQueueSize, alloc);
	/* @addition VAPG */
	if (m_image)  m_image->resize(getFrameSize());
	else m_image = alloc.new_object<Film>(getFrameSize());
	if (m_pixelEstimate)  m_pixelEstimate->resize(getFrameSize());
	else m_pixelEstimate = alloc.new_object<Film>(getFrameSize());
	m_image->reset();
	m_pixelEstimate->reset();
	m_task.reset();
	CUDA_SYNC_CHECK();
}

void PPGPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParameters<PPGPathTracer> params = {};
	params.traversable			  = backend->getRootTraversable();
	params.sceneData			  = backend->getSceneData();
	params.colorSpace			  = KRR_DEFAULT_COLORSPACE;
	params.currentRayQueue		  = currentRayQueue(depth);
	params.missRayQueue			  = missRayQueue;
	params.hitLightRayQueue		  = hitLightRayQueue;
	params.scatterRayQueue		  = scatterRayQueue;
	params.nextRayQueue			  = nextRayQueue(depth);
	params.mediumSampleQueue	  = enableMedium ? mediumSampleQueue : nullptr;
	params.pixelState			  = pixelState;
	backend->launch(params, "Closest", maxQueueSize, 1, 1, KRR_DEFAULT_STREAM);
}

void PPGPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParameters<PPGPathTracer> params = {};
	params.traversable			  = backend->getRootTraversable();
	params.sceneData			  = backend->getSceneData();
	params.colorSpace			  = KRR_DEFAULT_COLORSPACE;
	params.shadowRayQueue		  = shadowRayQueue;
	params.pixelState			  = pixelState;
	params.guidedState			  = guidedPathState;
	params.enableTraining		  = enableLearning;
	if(enableMedium) backend->launch(params, "ShadowTr", maxQueueSize, 1, 1, KRR_DEFAULT_STREAM);
	else backend->launch(params, "Shadow", maxQueueSize, 1, 1, KRR_DEFAULT_STREAM);
}

void PPGPathTracer::handleHit() {
	PROFILE("Process intersected rays");
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
			Spectrum Le =
				w.light.L(w.p, w.n, w.uv, w.wo, pixelState->lambda[w.pixelId]) * w.thp;
			if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
				Interaction intr(w.p, w.wo, w.n, w.uv);
				float lightPdf = w.light.pdfLi(intr, w.ctx) * lightSampler.pdf(w.light);
				Le /= (w.pl * lightPdf + w.pu).mean();
			} else Le /= w.pu.mean();
			pixelState->addRadiance(w.pixelId, Le);
			guidedPathState->recordRadiance(w.pixelId, Le);
	});
}

void PPGPathTracer::handleMiss() {
	PROFILE("Process escaped rays");
	const rt::SceneData& sceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(missRayQueue, maxQueueSize, KRR_DEVICE_LAMBDA(const MissRayWorkItem & w) {
			Spectrum L = {};
			const SampledWavelengths& lambda = pixelState->lambda[w.pixelId];
			Interaction intr(w.ray.origin);
			for (const rt::InfiniteLight &light : sceneData.infiniteLights) {
				if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
					float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
					L += light.Li(w.ray.dir, lambda) / (w.pu + w.pl * lightPdf).mean();
				} else L += light.Li(w.ray.dir, lambda) / w.pu.mean();	
			}
			pixelState->addRadiance(w.pixelId, w.thp * L);
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
		
		const SurfaceInteraction& intr = w.intr;
		BSDF bsdf(intr);
		BSDFType bsdfType = intr.getBsdfType();
		Vector3f woLocal = intr.toLocal(intr.wo);
		const SampledWavelengths& lambda	   = pixelState->lambda[w.pixelId];
		
		/* Statistics for mixed bsdf-guided sampling */
		float bsdfPdf, dTreePdf;
		DTreeWrapper* dTree = m_sdTree->dTreeWrapper(intr.p);

		if (enableNEE && (bsdfType & BSDF_SMOOTH)) {
			SampledLight sampledLight = lightSampler.sample(sampler.get1D());
			Light light				  = sampledLight.light;
			LightSample ls			  = light.sampleLi(sampler.get2D(), {intr.p, intr.n}, lambda);
			Ray shadowRay			  = intr.spawnRayTo(ls.intr);
			Vector3f wiWorld		  = normalize(shadowRay.dir);
			Vector3f wiLocal		  = intr.toLocal(wiWorld);

			float lightPdf	= sampledLight.pdf * ls.pdf;
			Spectrum bsdfVal	= bsdf.f(woLocal, wiLocal);
			float bsdfPdf	= light.isDeltaLight() ? 0 : bsdf.pdf(woLocal, wiLocal);
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

		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		guidedRayQueue->push(tid); /* unfinished rays */
	});
}

void PPGPathTracer::generateScatterRays(int depth) {
	PROFILE("Generate scatter rays");
	ForAllQueued(guidedRayQueue, maxQueueSize, 
		KRR_DEVICE_LAMBDA(const GuidedRayWorkItem &id) {
			const ScatterRayWorkItem w = scatterRayQueue->operator[](id.itemId);
			Sampler sampler				   = &pixelState->sampler[w.pixelId];
			const SurfaceInteraction &intr = w.intr;
			const BSDFType bsdfType		   = intr.getBsdfType();
			Vector3f woLocal			   = intr.toLocal(intr.wo);

			float bsdfPdf, dTreePdf;
			Vector3f dTreeVoxelSize{};
			DTreeWrapper *dTree = m_sdTree->dTreeWrapper(intr.p, dTreeVoxelSize);

			BSDFSample sample = PPGPathTracer::sample(sampler, intr, bsdfPdf, dTreePdf, w.depth,
													  m_bsdfSamplingFraction, dTree, bsdfType);
			if (sample.pdf > 0 && sample.f.any()) {
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
				if (any(r.thp)) {
					nextRayQueue(depth)->push(r);
					/* guidance... */
					if (r.depth <= MAX_TRAIN_DEPTH) {
						guidedPathState->incrementDepth(r.pixelId, r.ray, dTree, dTreeVoxelSize,
														r.thp, sample.f, sample.pdf, bsdfPdf, dTreePdf,
														sample.isDelta());
					}
				}
			}
	});
}

void PPGPathTracer::render(RenderContext *context) {
	if (!mScene || !maxQueueSize) return;
	PROFILE("PPG Path Tracer");
	CHECK_LOG(samplesPerPixel == 1, "Only 1spp/frame is supported for PPG!");
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
		generateCameraRays();
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
				guidedRayQueue->reset();
				if (enableMedium) {
					mediumSampleQueue->reset();
					mediumScatterQueue->reset();
				}
			});
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			traceClosest(depth);
			if (enableMedium) sampleMediumInteraction(depth);	
			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleHit();
			handleMiss();
			// Break on maximum depth, but incorprate contribution from emissive hits.
			if (depth == maxDepth) break;
			if (enableMedium) sampleMediumScattering(depth);
			// [STEP#2.3] handle intersections and shadow rays
			handleIntersections();
			if (enableNEE) traceShadow();
			// [STEP#2.4] towards next bounce
			generateScatterRays(depth);
		}
	}
	// write results of the current frame...
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
	GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
		RGB L = pixelState->L[pixelId].toRGB(pixelState->lambda[pixelId],
				*KRR_DEFAULT_COLORSPACE_GPU) / samplesPerPixel;
		if (enableClamp) L.clamp(0, clampMax);
		m_image->put(RGBA(L, 1.f), pixelId);
		if (m_renderMode == RenderMode::Interactive)
			frameBuffer.write(RGBA(L, 1), pixelId);
		else if (m_renderMode == RenderMode::Offline)
			frameBuffer.write(m_image->getPixel(pixelId), pixelId);
	});
}

void PPGPathTracer::beginFrame(RenderContext* context) {
	if (!mScene || !maxQueueSize) return;
	WavefrontPathTracer::beginFrame(context);
	GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		guidedPathState->n_vertices[pixelId] = 0;
	});
	// [offline mode] always training when auto-train enabled
	// but the last iteration (the render iteration) do not need training anymore.
	enableLearning = (enableLearning || m_autoBuild) && !m_isFinalIter;	
	train_frames_this_iteration = (1 << m_iter) * m_sppPerPass;
}

void PPGPathTracer::endFrame(RenderContext* context) {
	if (enableLearning) {
		PROFILE("Training SD-Tree");
		GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
			Sampler sampler = &pixelState->sampler[pixelId];
			Spectrum pixelEstimate(0.5);
			if (m_isBuilt && m_distribution == EDistribution::EFull)
				pixelEstimate = Spectrum::fromRGB(
					m_pixelEstimate->getPixel(pixelId).head<3>(), SpectrumType::RGBUnbounded,
					pixelState->lambda[pixelId], *KRR_DEFAULT_COLORSPACE_GPU);
				guidedPathState->commitAll(pixelId, m_sdTree, 1.f, m_spatialFilter,
										   m_directionalFilter, m_bsdfSamplingFractionLoss, sampler,
										   m_distribution, pixelState->lambda[pixelId],
										   pixelEstimate);
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
	if (mScene->getMedia().size()) 
		if (ui::Checkbox("Enable medium", &enableMedium)) initialize();

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
	if (enableClamp) ui::DragFloat("Max:", &clampMax, 1, 500);
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
	// TODO: implement weight by inverse variance.
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
	Log(Success, "Starting iteration #%d", m_iter);

}

void PPGPathTracer::finalize() { 
	if (m_renderMode == RenderMode::Offline) {
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
}

KRR_CALLABLE BSDFSample PPGPathTracer::sample(Sampler& sampler, 
	const SurfaceInteraction& intr, float& bsdfPdf, float& dTreePdf, int depth,
	float bsdfSamplingFraction, const DTreeWrapper* dTree, BSDFType bsdfType) const {
	BSDFSample sample = {};
	Vector3f woLocal = intr.toLocal(intr.wo);
	BSDF bsdf(intr);

	if (!m_isBuilt || !dTree || !enableGuiding || (bsdfType & BSDF_SPECULAR)
		|| bsdfSamplingFraction == 1 || depth >= MAX_GUIDED_DEPTH) {
		sample	 = bsdf.sample(woLocal, sampler);
		bsdfPdf	 = sample.pdf;
		dTreePdf = 0;
		return sample;
	}

	if (bsdfSamplingFraction > 0 && sampler.get1D() < bsdfSamplingFraction) {
		sample	   = bsdf.sample(woLocal, sampler);
		bsdfPdf	   = sample.pdf;
		dTreePdf   = dTree->pdf(intr.toWorld(sample.wi));
		sample.pdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
		return sample;
	}
	else {
		sample.wi	 = intr.toLocal(dTree->sample(sampler));
		sample.f	 = bsdf.f(woLocal, sample.wi);
		sample.flags = BSDF_GLOSSY | (SameHemisphere(sample.wi, woLocal) ?
			BSDF_REFLECTION : BSDF_TRANSMISSION);
		sample.pdf = evalPdf(bsdfPdf, dTreePdf, depth, intr, sample.wi,
			bsdfSamplingFraction, dTree, sample.flags /*The bsdf lobe type is needed (in case for delta lobes)*/);
		return sample;
	}
}

KRR_CALLABLE float PPGPathTracer::evalPdf(float& bsdfPdf, float& dTreePdf, int depth,
	const SurfaceInteraction& intr, Vector3f wiLocal, float alpha, const DTreeWrapper* dTree, BSDFType bsdfType) const {
	Vector3f woLocal = intr.toLocal(intr.wo);
	BSDF bsdf(intr);
	bsdfPdf = dTreePdf = 0;
	if (!m_isBuilt || !dTree || !enableGuiding || (bsdfType & BSDF_SPECULAR)
		|| alpha == 1 || depth >= MAX_GUIDED_DEPTH) {
		return bsdfPdf = bsdf.pdf(woLocal, wiLocal);
	}
	if (alpha > 0) {
		bsdfPdf = bsdf.pdf(woLocal, wiLocal);
		if (isinf(bsdfPdf) || isnan(bsdfPdf)) {
			return bsdfPdf = dTreePdf = 0;
		}
	}
	if (alpha < 1) {
		dTreePdf = dTree->pdf(intr.toWorld(wiLocal));
	}
	return alpha * bsdfPdf + (1 - alpha) * dTreePdf;
}

/* [At the begining of each iteration]
	Adaptively subdivide the S-Tree,
	and resets the distribution within the (building) D-Tree.
	The irradiance records within D-Tree is cleared after this. */
void PPGPathTracer::resetSDTree() {
	cudaDeviceSynchronize();
	/* About 18k at the first iteration. */
	float sTreeSplitThres = sqrt(pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold;
	Log(Info, "Adaptively subdividing the S-Tree. Current split threshould: %.2f", sTreeSplitThres);
	m_sdTree->refine((size_t) sTreeSplitThres, m_sdTreeMaxMemory);
	CUDA_SYNC_CHECK();
	Log(Info, "Adaptively subdividing the D-Tree...");
	float dTreeThres = this->m_dTreeThreshold;
	m_sdTree->forEachDTreeWrapper(
		[dTreeThres](DTreeWrapper *dTree) { dTree->reset(20 /* max d-tree depth */, dTreeThres); });
	CUDA_SYNC_CHECK();
}

/* [At the end of each iteration] Build the sampling distribution with statistics, in the current
 * iteration of the building tree, */
/* Then use it, on the sampling tree, in the next iteration. */
void PPGPathTracer::buildSDTree() {
	CUDA_SYNC_CHECK();
	// Build distributions
	Log(Info, "Building distributions for each D-Tree node...");
	EDistribution dist = m_distribution;
	m_sdTree->forEachDTreeWrapper([dist](DTreeWrapper *dTree) { dTree->build(dist); });
	m_isBuilt = true;
	CUDA_SYNC_CHECK();
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
	constexpr int SPECTRUM_SAMPLES	= Spectrum::dim;

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
NAMESPACE_END(krr)