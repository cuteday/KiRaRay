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
	static constexpr int MAX_DEPTH = 9;
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

	float evalPdf(float& bsdfPdf, float& dTreePdf,
		const ShadingData& sd, vec3f wi, float alpha, const DTreeWrapper* dTree) const;
	BSDFSample sample(Sampler sampler, const ShadingData& sd,
		float& bsdfPdf, float& dTreePdf,
		float bsdfSamplingFraction, const DTreeWrapper* dTree) const;

	void resetSDTree();
	void buildSDTree();

	STree* m_sdTree{};
	bool m_isBuilt{};
	int m_iter{};
	int m_sppPerPass{};
	int m_sTreeThreshold{};
	int m_dTreeThreshold{};
	int m_sdTreeMaxMemory{};
	float m_bsdfSamplingFraction{ 0.5 };
	EDirectionalFilter m_directionalFilter{ EDirectionalFilter::ENearest };
	ESpatialFilter m_spatialFilter{ ESpatialFilter::ENearest };
	EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss{ EBsdfSamplingFractionLoss::ENone };
	GuidedPathStateBuffer* guidedPathState;
};

KRR_NAMESPACE_END