#pragma once

#include "common.h"
#include "tree.h"
#include "render/wavefront/integrator.h"
#include "backend.h"

KRR_NAMESPACE_BEGIN

/* A minimized ppg on interactive gpu pathtracing, the simplifications are:
*   Sample without NEE;
*   No need to learn MIS probability;
*   No spatial or directional filters;
*   Manually reset and rebuild the sdtree (instead of automatically iterative learning);
*   No combining rendered frames (with optimal variance).
*/
class PPGPathTracer : public WavefrontPathTracer {
public:
	using SharedPtr = std::shared_ptr<PPGPathTracer>;
	KRR_REGISTER_PASS_DEC(PPGPathTracer);

	PPGPathTracer() = default;
	~PPGPathTracer() = default;
	void initialize();

	void resize(const Vector2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(CUDABuffer& frame) override;
	void endFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

	string getName() const override { return "PPGPathTracer"; }

	void handleHit();
	void handleMiss();
	void handleIntersections();
	void generateScatterRays();

	/* params:
	 *	wi: scatter direction in local shading frame
	 */
	KRR_CALLABLE float evalPdf(float& bsdfPdf, float& dTreePdf, int depth,
		const ShadingData& sd, Vector3f wiLocal, float alpha, const DTreeWrapper* dTree, BSDFType bsdfType = BSDF_UNSET) const;
	KRR_CALLABLE BSDFSample sample(Sampler& sampler, const ShadingData& sd,
		float& bsdfPdf, float& dTreePdf, int depth,
		float bsdfSamplingFraction, const DTreeWrapper* dTree, BSDFType bsdfType = BSDF_UNSET) const;

	/* This adaptively changes the topology for the S-Tree and D-Tree (the building tree). 
		This also resets the statistics within all D-Tree nodes (for the building tree). */
	void resetSDTree();
	/* This builds the sampling distribution for importance sampling, after collecting enough MC estimates. */
	void buildSDTree();

	GuidedRayQueue *guidedRayQueue;
	OptiXPPGBackend* backend;

	STree* m_sdTree{ 0 };
	bool m_isBuilt{ 0 };							/* Whether the sampling tree has been built at least once. */
	int m_iter{ 0 };								/* How many iteration have passed? Each iteration, the pass should be doubled. */
	int m_sppPerPass{ 10 };							/* A "pass" is some number of frames. */
	int m_sdTreeMaxMemory{ 16 };					/* Max memory in megabytes. */
	float m_bsdfSamplingFraction{ 0.5 };			
	int m_sTreeThreshold{ 12000 };					/* The subdivision threshold for the statistical weight of the S-Tree. */
	float m_dTreeThreshold{ 0.01 };					/* The subdivision / prune threshold for the D-Tree (the energy fraction of spherical area). */
	EDirectionalFilter m_directionalFilter{ EDirectionalFilter::ENearest };
	ESpatialFilter m_spatialFilter{ ESpatialFilter::ENearest };
	EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss{ EBsdfSamplingFractionLoss::ENone };
	
	bool enableLearning{false};
	bool enableGuiding{true};
	GuidedPathStateBuffer* guidedPathState{};

	friend void to_json(nlohmann::json &j, const PPGPathTracer &p) {
		nlohmann::to_json(j, static_cast<const WavefrontPathTracer&>(p));
		j.update({ 
			{ "spp_per_pass", p.m_sppPerPass },
			{ "max_memory", p.m_sdTreeMaxMemory }, 
			{ "bsdf_fraction", p.m_bsdfSamplingFraction }
		});
	}

	friend void from_json(const nlohmann::json &j, PPGPathTracer &p) {
		nlohmann::from_json(j, static_cast<WavefrontPathTracer &>(p));
		j.at("spp_per_pass").get_to(p.m_sppPerPass);
		j.at("max_memory").get_to(p.m_sdTreeMaxMemory);
		j.at("bsdf_fraction").get_to(p.m_bsdfSamplingFraction);
	}
};

KRR_NAMESPACE_END