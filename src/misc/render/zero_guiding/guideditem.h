#pragma once
#include "common.h"
#include "render/wavefront/workitem.h"
#include "render/wavefront/workqueue.h"
#include "ppg/tree.h"

KRR_NAMESPACE_BEGIN

constexpr int MAX_GUIDED_DEPTH = 10;
constexpr int MAX_TRAIN_DEPTH = 10;

struct GuidedRayWorkItem {
	uint itemId;	// the index of work item in scatter ray queue
};

struct Vertex {
	DTreeWrapper* dTree;
	Vector3f dTreeVoxelSize;
	Ray ray;
	SampledSpectrum throughput;
	SampledSpectrum bsdfVal;
	SampledSpectrum radiance;
	float wiPdf, bsdfPdf, dTreePdf;
	float wiMisWeight;	// @addition VAPG
	bool isDelta;

	KRR_DEVICE void record(const SampledSpectrum &r) {
		radiance += r;
	}

	KRR_DEVICE void commit(STree *sdTree, float statisticalWeight, ESpatialFilter spatialFilter,
						   EDirectionalFilter directionalFilter,
						   EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler &sampler,
						   EDistribution distribution = EDistribution::ERadiance,
						   const SampledSpectrum &pixelEstimate = {}) {
		if (wiPdf <= 0 || isDelta) return;
		
		SampledSpectrum localRadiance(0);
		if (throughput[0] * wiPdf > 1e-4f)
			localRadiance[0] = radiance[0] / throughput[0];
		if (throughput[1] * wiPdf > 1e-4f)
			localRadiance[1] = radiance[1] / throughput[1];
		if (throughput[2] * wiPdf > 1e-4f)
			localRadiance[2] = radiance[2] / throughput[2];
		SampledSpectrum product = localRadiance * bsdfVal;
		
		/* @modified: VAPG */
		float value	= localRadiance.mean();
		switch (distribution) {
			case EDistribution::ERadiance:
				value = localRadiance.mean();
				break;
			case EDistribution::ESimple:	/* consider partial integrand (additional BSDF) */
				value = product.mean();
				if (wiMisWeight > 0) value *= wiMisWeight;	// MIS aware
				value = pow2(value);		/* second moment */
				break;
			case EDistribution::EFull:
				// note that since 'radiance = throughput * localRadiance',
				// we do not need a multiplication by throughput here.
				// The 'pixelEstimate' is filtered and offseted so no nan will occur.
				value = (radiance / pixelEstimate * wiPdf).mean();	// full integrand
				if (wiMisWeight > 0) value *= wiMisWeight;	// MIS aware
				value = pow2(value);		/* second moment */
				break;
		}

		DTreeRecord rec{ray.dir, value,	   product.mean(),	  wiPdf,
						bsdfPdf, dTreePdf, statisticalWeight, isDelta};

		switch (spatialFilter) {
		case ESpatialFilter::ENearest:
			dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
			break;
		case ESpatialFilter::EStochasticBox:
		{
			DTreeWrapper* splatDTree = dTree;

			// Jitter the actual position within the
			// filter box to perform stochastic filtering.
			Vector3f offset = dTreeVoxelSize;
			offset[0] *= sampler.get1D() - 0.5f;
			offset[1] *= sampler.get1D() - 0.5f;
			offset[2] *= sampler.get1D() - 0.5f;

			const AABB& sdAabb = sdTree->aabb();
			Vector3f origin	   = sdAabb.clip(ray.origin + offset);

			splatDTree = sdTree->dTreeWrapper(origin);
			if (splatDTree) {
				splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
			}
			break;
		}
		case ESpatialFilter::EBox:
			sdTree->record(ray.origin, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
			break;
		}
	}
};

struct GuidedPathState {
	Vertex vertices[MAX_TRAIN_DEPTH];
	uint n_vertices{};
};

#include "zero_guiding/guideditem_soa.h"

class GuidedRayQueue : public WorkQueue<GuidedRayWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(uint index) { return push(GuidedRayWorkItem{ index }); }
};

class GuidedPathStateBuffer : public SOA<GuidedPathState> {
public:
	GuidedPathStateBuffer() = default;
	GuidedPathStateBuffer(int n, Allocator alloc) : SOA<GuidedPathState>(n, alloc) {}

	KRR_DEVICE void incrementDepth(int pixelId, Ray &ray, DTreeWrapper *dTree,
								   Vector3f dTreeVoxelSize, SampledSpectrum thp,
								   SampledSpectrum bsdfVal, float wiPdf, float bsdfPdf,
								   float dTreePdf, bool isDelta = false) {
		int depth = n_vertices[pixelId];
		if (depth >= MAX_TRAIN_DEPTH) return;
		/* Always remember to clear the radiance from last frame, as I forgotten... */
		vertices[depth].radiance[pixelId] = SampledSpectrum(0); 
		vertices[depth].ray[pixelId]			= ray;
		vertices[depth].dTree[pixelId]			= dTree;
		vertices[depth].dTreeVoxelSize[pixelId] = dTreeVoxelSize;
		vertices[depth].throughput[pixelId]		= thp;
		vertices[depth].bsdfVal[pixelId]		= bsdfVal;
		vertices[depth].wiPdf[pixelId]			= wiPdf;
		vertices[depth].bsdfPdf[pixelId]		= bsdfPdf;
		vertices[depth].dTreePdf[pixelId]		= dTreePdf;
		vertices[depth].isDelta[pixelId]		= isDelta;
		vertices[depth].wiMisWeight[pixelId] =
			dTreePdf / wiPdf; // currently ok since we used a constant selction prob
		n_vertices[pixelId] = 1 + depth;
	}

	KRR_DEVICE void recordRadiance(int pixelId, SampledSpectrum L) {
		int cur_depth = n_vertices[pixelId];
		for (int i = 0; i < cur_depth; i++) {
			SampledSpectrum prevRadiance  = vertices[i].radiance[pixelId];
			vertices[i].radiance[pixelId] = prevRadiance + L;
		}
	}

	// @modified VAPG
	KRR_DEVICE void commitAll(int pixelId, 
		STree* sdTree, float statisticalWeight,
		ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter,
		EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler& sampler,
		EDistribution distribution = EDistribution::ERadiance, const SampledSpectrum &pixelEstimate = {}) {
		for (int i = 0; i < n_vertices[pixelId]; i++) {
			Vertex v = vertices[i][pixelId];
			v.commit(sdTree, statisticalWeight,
				spatialFilter, directionalFilter, bsdfSamplingFractionLoss, sampler, 
				distribution, pixelEstimate);
		}
	}
};

KRR_NAMESPACE_END