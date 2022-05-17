#include "common.h"
#include "window.h"
#include "math/math.h"

#include "sdtree.h"
#include "guided_pt.h"

KRR_NAMESPACE_BEGIN

void GuidedPathTracer::resize(const vec2i& size) {
	frameSize = size;
	initialize();		// need to resize the queues
}

void GuidedPathTracer::setScene(Scene::SharedPtr scene) {
	scene->toDevice();
	mpScene = scene;
	lightSampler = scene->getSceneData().lightSampler;
	initialize();
	backend->setScene(*scene);
}

void GuidedPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	WavefrontPathTracer::initialize();
	CUDA_SYNC_CHECK();
	if (guidedPathState) guidedPathState->resize(maxQueueSize, alloc);
	else guidedPathState = alloc.new_object<GuidedPathStateBuffer>(maxQueueSize, alloc);
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::handleHit() {
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
		pixelState->addRadiance(w.pixelId, Le* w.thp* misWeight);
		guidedPathState->recordRadiance(w.pixelId, Le * w.thp * misWeight);
	});
}

void GuidedPathTracer::handleMiss() {
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem & w) {
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
		pixelState->addRadiance(w.pixelId, w.thp* L);
		guidedPathState->recordRadiance(w.pixelId, w.thp * L);	// ppg
	});
}

void GuidedPathTracer::generateScatterRays() {
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
			guidedPathState->incrementDepth(r.pixelId, r.thp, sample.f);
		}
	});
}

void GuidedPathTracer::render(CUDABuffer& frame) {
	if (!mpScene || !maxQueueSize) return;
	vec4f* frameBuffer = (vec4f*)frame.data();
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		Call(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; depth < min(maxDepth, MAX_GUIDED_DEPTH); depth++) {
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
	CUDA_SYNC_CHECK();
	frameId++;
}

void GuidedPathTracer::beginFrame(CUDABuffer& frame) {
	if (!mpScene || !maxQueueSize) return;
	WavefrontPathTracer::beginFrame(frame);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		guidedPathState->n_vertices[pixelId] = 0;
	});
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::endFrame(CUDABuffer& frame){
	vec4f* frameBuffer = (vec4f*)frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		vec3f L = vec3f(pixelState->L[pixelId]) / samplesPerPixel;
		if (enableClamp) L = clamp(L, vec3f(0), vec3f(clampMax));
		frameBuffer[pixelId] = vec4f(L, any(L) ? 1 : 0);
	});
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Sampler sampler = &pixelState->sampler[pixelId];
		guidedPathState->commitAll(pixelId, *m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter,
			m_bsdfSamplingFractionLoss, sampler);
	});
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::renderUI() {
	if (ui::CollapsingHeader("Wavefront path tracer")) {
		ui::Text("Hello from wavefront path tracer!");
		ui::InputInt("Samples per pixel", &samplesPerPixel);
		ui::SliderInt("Max recursion depth", &maxDepth, 0, MAX_GUIDED_DEPTH);
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
			ui::SliderFloat("Max:", &clampMax, 1, 500);
		}
		ui::Text("Misc");
		ui::Checkbox("Transparent background", &transparentBackground);
	}
}


KRR_CALLABLE BSDFSample GuidedPathTracer::sample(Sampler sampler, const ShadingData& sd,
	float& bsdfPdf, float& dTreePdf, 
	float bsdfSamplingFraction, const DTreeWrapper* dTree) const {
	vec2f u = sampler.get2D();
	BSDFSample sample = {};
	vec3f woLocal = sd.toLocal(sd.wo);

	if (!m_isBuilt || !dTree) {
		sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		bsdfPdf = sample.pdf;
		dTreePdf = 0;
		return sample;
	}

	vec3f result;
	if (u.x < bsdfSamplingFraction) {
		u.x /= bsdfSamplingFraction;
		sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		if (!any(sample.f)) {
			bsdfPdf = dTreePdf = 0;
			return sample;
		}
	}
	else {
		u.x = (u.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);
		sample.wi = sd.toLocal(dTree->sample(sampler));
		sample.f = BxDF::f(sd, woLocal, sample.wi, (int)sd.bsdfType);
	}

	sample.pdf = evalPdf(bsdfPdf, dTreePdf, 
		sd, sample.wi, bsdfSamplingFraction, dTree);
	if (sample.pdf == 0) sample.f = 0;
	return sample;
}

KRR_CALLABLE float GuidedPathTracer::evalPdf(float& bsdfPdf, float& dTreePdf,
	const ShadingData& sd, vec3f wi, float alpha, const DTreeWrapper* dTree) const {
	vec3f wo = sd.wo;
	if (!m_isBuilt || !dTree) {
		dTreePdf = 0;
		return bsdfPdf = BxDF::pdf(sd, wo, wi, (int)sd.bsdfType);
	}
	bsdfPdf = BxDF::pdf(sd, wo, wi, (int)sd.bsdfType);
	if (!isinf(bsdfPdf)) {
		return 0;
	}
	dTreePdf = dTree->pdf(sd.fromLocal(wo));
	return alpha * bsdfPdf + (1 - alpha) * dTreePdf;
}

void GuidedPathTracer::resetSDTree() {
	logInfo("Resetting distributions for sampling.");

	m_sdTree->refine((size_t)(std::sqrt(std::pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold), m_sdTreeMaxMemory);
	m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(20, m_dTreeThreshold); });
}

void GuidedPathTracer::buildSDTree() {
	logInfo("Building distributions for sampling.");

	// Build distributions
	m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });

	// Gather statistics
	int maxDepth = 0;
	int minDepth = std::numeric_limits<int>::max();
	float avgDepth = 0;
	float maxAvgRadiance = 0;
	float minAvgRadiance = std::numeric_limits<float>::max();
	float avgAvgRadiance = 0;
	size_t maxNodes = 0;
	size_t minNodes = std::numeric_limits<size_t>::max();
	float avgNodes = 0;
	float maxStatisticalWeight = 0;
	float minStatisticalWeight = std::numeric_limits<float>::max();
	float avgStatisticalWeight = 0;

	int nvec3fs = 0;
	int nvec3fsNodes = 0;

	m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
		const int depth = dTree->depth();
		maxDepth = std::max(maxDepth, depth);
		minDepth = std::min(minDepth, depth);
		avgDepth += depth;

		const float avgRadiance = dTree->meanRadiance();
		maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
		minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
		avgAvgRadiance += avgRadiance;

		if (dTree->numNodes() > 1) {
			const size_t nodes = dTree->numNodes();
			maxNodes = std::max(maxNodes, nodes);
			minNodes = std::min(minNodes, nodes);
			avgNodes += nodes;
			++nvec3fsNodes;
		}

		const float statisticalWeight = dTree->statisticalWeight();
		maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
		minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
		avgStatisticalWeight += statisticalWeight;

		++nvec3fs;
		});

	if (nvec3fs > 0) {
		avgDepth /= nvec3fs;
		avgAvgRadiance /= nvec3fs;
		if (nvec3fsNodes > 0) {
			avgNodes /= nvec3fsNodes;
		}
		avgStatisticalWeight /= nvec3fs;
	}
}


KRR_NAMESPACE_END