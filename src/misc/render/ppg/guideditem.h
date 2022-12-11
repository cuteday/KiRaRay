#pragma once
#include "common.h"
#include "render/wavefront/workitem.h"
#include "render/wavefront/workqueue.h"
#include "tree.h"

KRR_NAMESPACE_BEGIN

constexpr int MAX_GUIDED_DEPTH = 5;
constexpr int MAX_TRAIN_DEPTH = 5;

struct GuidedRayWorkItem {
	uint itemId;	// the index of work item in scatter ray queue
};

struct Vertex {
	DTreeWrapper* dTree;
	Vector3f dTreeVoxelSize;
	Ray ray;
	Color3f throughput;
	Color3f bsdfVal;
	Color3f radiance;
	float woPdf, bsdfPdf, dTreePdf;
	bool isDelta;

	KRR_DEVICE void record(const Color3f& r) {
		radiance += r;
	}

	KRR_DEVICE void commit(STree* sdTree, float statisticalWeight,
		ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter,
		EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler& sampler) {
		if (woPdf <= 0 || isDelta) {
			return;
		}
		Color3f localRadiance = 0;
		if (throughput[0] * woPdf > 1e-4f) localRadiance[0] = radiance[0] / throughput[0];
		if (throughput[1] * woPdf > 1e-4f) localRadiance[1] = radiance[1] / throughput[1];
		if (throughput[2] * woPdf > 1e-4f) localRadiance[2] = radiance[2] / throughput[2];
		Color3f product = localRadiance * bsdfVal;

		//if (localRadiance.mean() / woPdf > 0)
		//	printf("Recording an seemingly unreasonbly large irradiance value %f: "
		//		"throughput [%f, %f, %f]; radiance [%f, %f, %f], woPdf %f\n",
		//		localRadiance.mean() / woPdf, throughput[0], throughput[1], throughput[2],
		//		   radiance[0], radiance[1], radiance[2], woPdf);

		DTreeRecord rec{ ray.dir, 
			localRadiance.mean(), product.mean(), 
			woPdf, bsdfPdf, dTreePdf, 
			statisticalWeight, 
			isDelta };

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
			Vector3f origin = ray.origin + offset;
			for (int i = 0; i < Vector3f::dim; i++) {
				origin[i] = max(sdAabb.min()[i], min(sdAabb.max()[i], origin[i]));
			}

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

#include "ppg/guideditem_soa.h"

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

	KRR_DEVICE void incrementDepth(int pixelId, 
		Ray& ray,
		DTreeWrapper* dTree,
		Vector3f dTreeVoxelSize,
		Color3f thp,
		Color3f bsdfVal,
		float woPdf, float bsdfPdf, float dTreePdf,
		bool isDelta = false) {
		int depth = n_vertices[pixelId];
		//printf("Attempting to increment depth of pixel %d with current depth %d\n",
		//	pixelId, depth);
		if (depth >= MAX_TRAIN_DEPTH) return;
		vertices[depth].radiance[pixelId]		= Color3f(0); /* Always remember to clear the radiance from last frame, as I forgotten... */
		vertices[depth].ray[pixelId]			= ray;
		vertices[depth].dTree[pixelId]			= dTree;
		vertices[depth].dTreeVoxelSize[pixelId] = dTreeVoxelSize;
		vertices[depth].throughput[pixelId]		= thp;
		vertices[depth].bsdfVal[pixelId]		= bsdfVal;
		vertices[depth].woPdf[pixelId]			= woPdf;
		vertices[depth].bsdfPdf[pixelId]		= bsdfPdf;
		vertices[depth].dTreePdf[pixelId]		= dTreePdf;
		vertices[depth].isDelta[pixelId]		= isDelta;
		n_vertices[pixelId] = 1 + depth;
	}

	KRR_DEVICE void recordRadiance(int pixelId, Color3f L) {
		int cur_depth = n_vertices[pixelId];
		for (int i = 0; i < cur_depth; i++) {
			Color3f prevRadiance = vertices[i].radiance[pixelId];
			vertices[i].radiance[pixelId] = prevRadiance + L;
			//if (L.mean() > 0 && prevRadiance.mean() > 0)
			//printf("Recording radiance %f for pixel #%d with current depth %d, current radiance: %f\n",
			//	L.mean(), pixelId, cur_depth, vertices[i].radiance[pixelId].mean());
		}
	}

	KRR_DEVICE void commitAll(int pixelId, 
		STree* sdTree, float statisticalWeight,
		ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter,
		EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler& sampler) {
		//printf("The current pixel #%d has %d vertices\n", pixelId, n_vertices[pixelId]);
		for (int i = 0; i < n_vertices[pixelId]; i++) {
			Vertex v = vertices[i][pixelId];
			v.commit(sdTree, statisticalWeight,
				spatialFilter, directionalFilter, bsdfSamplingFractionLoss, sampler);
		}
	}
};

KRR_NAMESPACE_END