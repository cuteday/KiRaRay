#pragma once

#include "common.h"
#include "sdtree.h"
#include "render/wavefront/integrator.h"
#include "guideditem.h"

KRR_NAMESPACE_BEGIN

/* Goal: a minimized ppg on interactive gpu pathtracing, the simplifications are:
*   Sample without NEE;
*   No need to learn MIS probability;
*   No spatial or directional filters;
*   Manually reset and rebuild the sdtree (instead of automatically iterative learning);
*   No combining rendered frames (with optimal variance).
*/
class GuidedPathTracer : public WavefrontPathTracer {
public:
	GuidedPathTracer() = default;
	~GuidedPathTracer() = default;
	void initialize();

	void resize(const vec2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(CUDABuffer& frame) override;
	void endFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

	void handleHit();
	void handleMiss();
	void generateScatterRays();

	KRR_CALLABLE float evalPdf(float& bsdfPdf, float& dTreePdf,
		const ShadingData& sd, vec3f wi, float alpha, const DTreeWrapper* dTree) const;
	KRR_CALLABLE BSDFSample sample(Sampler sampler, const ShadingData& sd,
		float& bsdfPdf, float& dTreePdf,
		float bsdfSamplingFraction, const DTreeWrapper* dTree) const;

	void resetSDTree();
	void buildSDTree();

	STree* m_sdTree{0};
	bool m_isBuilt{0};
	int m_iter{0};
	int m_sppPerPass{10};
	int m_sdTreeMaxMemory{64};
	float m_bsdfSamplingFraction{ 0.5 };	
	int m_sTreeThreshold{ 12000 };
	float m_dTreeThreshold{ 0.01 };
	EDirectionalFilter m_directionalFilter{ EDirectionalFilter::ENearest };
	ESpatialFilter m_spatialFilter{ ESpatialFilter::ENearest };
	EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss{ EBsdfSamplingFractionLoss::ENone };
	
	bool enableLearning{true};
	bool enableGuiding{true};
	GuidedPathStateBuffer* guidedPathState{};
};

KRR_NAMESPACE_END