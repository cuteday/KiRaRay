#pragma once
#include "common.h"
#include "taggedptr.h"
#include "render/phase.h"
#include "medium.h"

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

KRR_NAMESPACE_BEGIN

class Ray;

struct MediumProperties {
	Color sigma_a, sigma_s;
	PhaseFunction phase;
	Color Le;
};

struct RayMajorantSegment {
	float tMin, tMax;
	Color sigma_maj;
};

class RayMajorantIterator {
public:
	KRR_CALLABLE RayMajorantIterator() = default;

	KRR_CALLABLE RayMajorantIterator(float tMin, float tMax, const Color &sigma_maj) :
		tMin(tMin), tMax(tMax), sigma_maj(sigma_maj) {}

	KRR_CALLABLE RayMajorantSegment next() { return {tMin, tMax, sigma_maj}; }

	float tMin, tMax;
	Color sigma_maj;
};

class HomogeneousMedium {
public:
	HomogeneousMedium(Color sigma_a, Color sigma_s, Color L_e, float g) :
		sigma_a(sigma_a), sigma_s(sigma_s), L_e(L_e), phase(g) {}

	KRR_CALLABLE bool isEmissive() const { return !L_e.isZero(); }

	KRR_CALLABLE Color Le(const Vector3f &p) const { return L_e; }
	
	KRR_CALLABLE MediumProperties samplePoint(Vector3f p) const {
		return {sigma_a, sigma_s, &phase, L_e};
	}

	KRR_CALLABLE RayMajorantIterator sampleRay(const Ray &ray, float tMax) {
		return {0, tMax, sigma_a + sigma_s};
	}

	Color sigma_a, sigma_s, L_e;
	HGPhaseFunction phase;
};

class NanoVDBMedium {
public:
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;

	NanoVDBMedium(const Affine3f& transform, Color sigma_a, Color sigma_s, Color sigma_maj,
		Color Le, float g, nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> gridHandle) :
		transform(transform), phase(g), L_e(Le),
		sigma_a(sigma_a), sigma_s(sigma_s), sigma_maj(sigma_maj) {

		inverseTransform = transform.inverse();
		densityGrid		 = gridHandle.grid<float>();
		nanovdb::BBox<nanovdb::Vec3R> bbox = densityGrid->worldBBox();
		bounds = AABB3f{Vector3f{bbox.min()[0], bbox.min()[1], bbox.min()[2]}, 
			Vector3f{bbox.max()[0], bbox.max()[1], bbox.max()[2]}};
	}

	KRR_CALLABLE bool isEmissive() const { return !L_e.isZero(); }

	KRR_CALLABLE Color Le(const Vector3f &p) const { return L_e; }

	KRR_CALLABLE MediumProperties samplePoint(Vector3f p) const { 
		p = inverseTransform * p;
		nanovdb::Vec3<float> pIndex = densityGrid->worldToIndexF(nanovdb::Vec3<float>(p.x(), p.y(), p.z()));
		float d = VDBSampler(densityGrid->tree())(pIndex);
		return {sigma_a * d, sigma_s * d, &phase, Le(p)};
	}

	KRR_CALLABLE RayMajorantIterator sampleRay(const Ray &ray, float raytMax);

	AABB3f bounds;
	Affine3f transform, inverseTransform;
	HGPhaseFunction phase;
	//nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> densityGridHandle;
	nanovdb::FloatGrid *densityGrid;
	Color sigma_a, sigma_s, sigma_maj;
	Color L_e;
};

class MediumInterface {
public:
	KRR_CALLABLE MediumInterface() = default;
	KRR_CALLABLE MediumInterface(Medium m) : inside(m), outside(m) {}
	KRR_CALLABLE MediumInterface(Medium mi, Medium mo) : inside(mi), outside(mo) {}

	KRR_CALLABLE bool isTransition() const { return inside != outside; }

	Medium inside, outside;
};

KRR_NAMESPACE_END