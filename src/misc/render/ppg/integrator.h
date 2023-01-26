#pragma once

#include "common.h"
#include "tree.h"
#include "render/wavefront/integrator.h"
#include "backend.h"
#include "util/task.h"

KRR_NAMESPACE_BEGIN

class Film;

class PPGPathTracer : public WavefrontPathTracer{
public:
	using SharedPtr = std::shared_ptr<PPGPathTracer>;
	KRR_REGISTER_PASS_DEC(PPGPathTracer);
	enum class RenderMode { Interactive, Offline };

	PPGPathTracer() = default;
	~PPGPathTracer() = default;
	void initialize();

	void resize(const Vector2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(CUDABuffer& frame) override;
	void endFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;
	void finalize() override; /* Save the rendering (of the last iter) maybe more. */

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
	/* @addition VAPG filter the raw pixel estimate using a simple box filter. */
	void filterFrame(Film *image);

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
	void nextIteration();								/* Do the works for entering NEXT, e.g., rebuild, save image */
	void resetGuiding();								/* Reset the SD-Tree to the beginning. */
	RenderMode m_renderMode{RenderMode::Interactive};	/* If in OFFLINE mode, most of the operations is automatic.  */	
	int m_trainingIterations{ -1 };						/* The number of iterations for training (-1 means depends on budget) */
	bool m_autoBuild{ false };							/* Automatically rebuild if the current render pass finishes. */
	bool m_isFinalIter{ false };						/* Only results of the final iter is saved */
	bool m_saveIntermediate{ false };					/* Save rendered images of each iteration. */
	RenderTask m_task{};								/* Task class for progressing and more */
	Film *m_image{ nullptr };							/* The image currently being rendered. @addition VAPG */
	Film *m_pixelEstimate{ nullptr };					/* The image rendered during the last iteration. @addition VAPG */
	
	EDirectionalFilter m_directionalFilter{ EDirectionalFilter::ENearest };
	ESpatialFilter m_spatialFilter{ ESpatialFilter::ENearest };
	EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss{ EBsdfSamplingFractionLoss::ENone };
	ESampleCombination m_sampleCombination{ ESampleCombination::EWeightBySampleCount };

	bool enableLearning{false};
	bool enableGuiding{true};
	GuidedPathStateBuffer* guidedPathState{};

	friend void to_json(json &j, const PPGPathTracer &p) {
		to_json(j, static_cast<const WavefrontPathTracer&>(p));
		j.update({ 
			{ "mode", p.m_renderMode },
			{ "target_dist",  p.m_distribution}, 
			{ "spp_per_pass", p.m_sppPerPass },
			{ "max_memory", p.m_sdTreeMaxMemory }, 
			{ "bsdf_fraction", p.m_bsdfSamplingFraction },
			{ "distribution", p.m_distribution },
			{ "spatial_filter", p.m_spatialFilter },
			{ "directional_filter", p.m_directionalFilter },
			{ "stree_thres", p.m_sTreeThreshold },
			{ "dtree_thres", p.m_dTreeThreshold },
			{ "auto_build", p.m_autoBuild },
			{ "budget", p.m_task },
			{ "save_intermediate", p.m_saveIntermediate },
			{ "training_iter", p.m_trainingIterations }
		});
	}

	friend void from_json(const json &j, PPGPathTracer &p) {
		from_json(j, static_cast<WavefrontPathTracer &>(p));
		p.m_renderMode			 = j.value("mode", PPGPathTracer::RenderMode::Interactive);		
		p.m_sppPerPass			 = j.value("spp_per_pass", 4);
		p.m_sdTreeMaxMemory		 = j.value("max_memory", 16);
		p.m_bsdfSamplingFraction = j.value("bsdf_fraction", 0.5);
		p.m_distribution		 = j.value("distribution", EDistribution::ERadiance);
		p.m_spatialFilter		 = j.value("spatial_filter", ESpatialFilter::ENearest);
		p.m_directionalFilter	 = j.value("directional_filter", EDirectionalFilter::ENearest);
		p.m_sampleCombination	 = j.value("sample_combination", ESampleCombination::EWeightBySampleCount); 
		p.m_sTreeThreshold		 = j.value("stree_thres", 4000.f);
		p.m_dTreeThreshold		 = j.value("dtree_thres", 0.01f);
		p.m_autoBuild			 = j.value("auto_build", false);
		p.enableGuiding			 = j.value("enable_guiding", true);
		p.m_task				 = j.value("budget", RenderTask{});
		p.m_saveIntermediate	 = j.value("save_intermediate", false);
		p.m_trainingIterations	 = j.value("training_iter", -1);
	}
};

KRR_ENUM_DEFINE(PPGPathTracer::RenderMode, {
	{ PPGPathTracer::RenderMode::Interactive, "interactive" },
	{PPGPathTracer::RenderMode::Offline, "offline"},
})

KRR_NAMESPACE_END