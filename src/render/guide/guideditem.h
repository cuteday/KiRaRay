#include "common.h"
#include "render/wavefront/workitem.h"

KRR_NAMESPACE_BEGIN

constexpr int MAX_GUIDED_DEPTH = 9;

struct Vertex {
	DTreeWrapper* dTree;
	vec3f dTreeVoxelSize;
	Ray ray;
	vec3f throughput;
	vec3f bsdfVal;
	vec3f radiance;
	float woPdf, bsdfPdf, dTreePdf;
	bool isDelta;

	KRR_CALLABLE void record(const vec3f& r) {
		radiance += r;
	}

	KRR_CALLABLE void commit(STree* sdTree, float statisticalWeight,
		ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter,
		EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler sampler) {
		// TODO check valid radiance and bsdfVal here...
		// if (woPdf <= 0 || !isnan(radiance) || !isnan(bsdfVal)) {
		if (woPdf <= 0) {
			return;
		}
		vec3f localRadiance = vec3f{ 0.0f };
		if (throughput[0] * woPdf > M_EPSILON) localRadiance[0] = radiance[0] / throughput[0];
		if (throughput[1] * woPdf > M_EPSILON) localRadiance[1] = radiance[1] / throughput[1];
		if (throughput[2] * woPdf > M_EPSILON) localRadiance[2] = radiance[2] / throughput[2];
		vec3f product = localRadiance * bsdfVal;

		DTreeRecord rec{ ray.dir, average(localRadiance), average(product), woPdf, bsdfPdf, dTreePdf, statisticalWeight, isDelta };
		switch (spatialFilter) {
		case ESpatialFilter::ENearest:
			dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
			break;
		case ESpatialFilter::EStochasticBox:
		{
			DTreeWrapper* splatDTree = dTree;

			// Jitter the actual position within the
			// filter box to perform stochastic filtering.
			vec3f offset = dTreeVoxelSize;
			offset.x *= sampler.get1D() - 0.5f;
			offset.y *= sampler.get1D() - 0.5f;
			offset.z *= sampler.get1D() - 0.5f;

			const AABB& sdAabb = sdTree->aabb();
			vec3f origin = ray.origin + offset;
			for (int i = 0; i < vec3f::dims; i++) {
				origin[i] = max(sdAabb.lower[i], min(sdAabb.upper[i], origin[i]));
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
	Vertex vertices[MAX_GUIDED_DEPTH];
	uint n_vertices{};
};

#include "guideditem_soa.h"

class GuidedPathStateBuffer : public SOA<GuidedPathState> {
public:
	GuidedPathStateBuffer() = default;
	GuidedPathStateBuffer(int n, Allocator alloc) : SOA<GuidedPathState>(n, alloc) {}

	KRR_CALLABLE void incrementDepth(int pixelId, 
		Ray& ray,
		DTreeWrapper* dTree,
		vec3f dTreeVoxelSize,
		vec3f thp,
		vec3f bsdfVal, 
		float woPdf, float bsdfPdf, float dTreePdf) {
		int depth = n_vertices[pixelId];
		vertices[depth].ray[pixelId] = ray;
		vertices[depth].dTree[pixelId] = dTree;
		vertices[depth].dTreeVoxelSize[pixelId] = dTreeVoxelSize;
		vertices[depth].throughput[pixelId] = thp;
		vertices[depth].bsdfVal[pixelId] = bsdfVal;
		vertices[depth].woPdf[pixelId] = woPdf;
		vertices[depth].bsdfPdf[pixelId] = bsdfPdf;
		vertices[depth].dTreePdf[pixelId] = dTreePdf;
		n_vertices[pixelId] = 1 + depth;
	}

	KRR_CALLABLE void recordRadiance(int pixelId, color L) {
		int cur_depth = n_vertices[pixelId];
		for (int i = 0; i < cur_depth; i++) {
			vec3f prevRadiance = vertices[i].radiance[pixelId];
			vertices[i].radiance[pixelId] = prevRadiance + L;
		}
	}

	KRR_CALLABLE void commitAll(int pixelId, 
		STree* sdTree, float statisticalWeight,
		ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter,
		EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler sampler) {
		for (int i = 0; i < n_vertices[pixelId]; i++) {
			Vertex v = vertices[i][pixelId];
			v.commit(sdTree, statisticalWeight,
				spatialFilter, directionalFilter, bsdfSamplingFractionLoss, sampler);
		}
	}
};

KRR_NAMESPACE_END