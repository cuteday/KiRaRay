#pragma once

#include "common.h"
#include "tree.h"
#include "render/wavefront/integrator.h"
#include "backend.h"

KRR_NAMESPACE_BEGIN

class Film;

/* A simplified ppg on interactive gpu pathtracing, the simplifications are:
*   No need to learn MIS probability;
*	No DI strategy (no guide towards direct illumination);
*   No spatial or directional filters;
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
	EDistribution m_distribution{ EDistribution::ERadiance };	/* The target distribution (radiance or radiance * bsdf). */
	bool m_isBuilt{ 0 };								/* Whether the sampling tree has been built at least once. */
	int m_iter{ 0 };									/* How many iteration have passed? Each iteration, the pass should be doubled. */
	int m_sppPerPass{ 10 };								/* A "pass" is some number of frames. */
	int m_sdTreeMaxMemory{ 16 };						/* Max memory in megabytes. */
	float m_bsdfSamplingFraction{ 0.5 };			
	int m_sTreeThreshold{ 12000 };						/* The subdivision threshold for the statistical weight of the S-Tree. */
	float m_dTreeThreshold{ 0.01 };						/* The subdivision / prune threshold for the D-Tree (the energy fraction of spherical area). */
	
	/* The following state parameters are used in offline setup with a given budget. */
	int m_trainingIterations{ -1 };						/* The number of iterations for training (-1 means unlimited) */
	bool m_isFinalIter{ false };						/* Only results of the final iter is saved */
	Film *m_image{ nullptr };							/* The image currently being rendered. @addition VAPG */
	Film *m_pixelEstimate{ nullptr };					/* The image rendered during the last iteration. @addition VAPG */

	EDirectionalFilter m_directionalFilter{ EDirectionalFilter::ENearest };
	ESpatialFilter m_spatialFilter{ ESpatialFilter::ENearest };
	EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss{ EBsdfSamplingFractionLoss::ENone };
	
	bool enableLearning{false};
	bool enableGuiding{true};
	GuidedPathStateBuffer* guidedPathState{};

	friend void to_json(json &j, const PPGPathTracer &p) {
		to_json(j, static_cast<const WavefrontPathTracer&>(p));
		j.update({ 
			{ "target_dist",  p.m_distribution}, 
			{ "spp_per_pass", p.m_sppPerPass },
			{ "max_memory", p.m_sdTreeMaxMemory }, 
			{ "bsdf_fraction", p.m_bsdfSamplingFraction },
			{ "distribution", p.m_distribution },
			{ "stree_thres", p.m_sTreeThreshold },
			{ "dtree_thres", p.m_dTreeThreshold }	
		});
	}

	friend void from_json(const json &j, PPGPathTracer &p) {
		//from_json(j, static_cast<WavefrontPathTracer &>(p));
		p.enableNEE				 = j.value("nee", true);
		p.maxDepth				 = j.value("max_depth", 6);
		p.probRR				 = j.value("rr", 0.8f);
		p.m_sppPerPass			 = j.value("spp_per_pass", 4);
		p.m_sdTreeMaxMemory		 = j.value("max_memory", 16);
		p.m_bsdfSamplingFraction = j.value("bsdf_fraction", 0.5);
		p.m_distribution		 = j.value("distribution", EDistribution::ERadiance);
		p.m_sTreeThreshold		 = j.value("stree_thres", 12000.f);
		p.m_dTreeThreshold		 = j.value("dtree_thres", 0.01f);
	}
};

KRR_NAMESPACE_END