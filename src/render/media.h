#pragma once
#include "common.h"
#include "taggedptr.h"
#include "render/phase.h"
#include "medium.h"
#include "raytracing.h"

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

KRR_NAMESPACE_BEGIN

struct MediumProperties {
	Color sigma_a, sigma_s;
	PhaseFunction phase;
	Color Le;
};

class RayMajorant {
public:
	KRR_CALLABLE RayMajorant() = default;

	KRR_CALLABLE RayMajorant(const Color &sigma_maj, 
		float tMin = 0, float tMax = M_FLOAT_INF) :
		sigma_maj(sigma_maj), tMin(tMin), tMax(tMax) {}

	float tMin, tMax;
	Color sigma_maj;
};

class NanoVDBGrid {
public:
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;

	NanoVDBGrid(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> densityHandle,
		float maxDensity) : densityHandle(std::move(densityHandle)), maxDensity(maxDensity) {
		densityGrid						   = densityHandle.grid<float>();
		nanovdb::BBox<nanovdb::Vec3R> bbox = densityGrid->worldBBox();
		bounds = AABB3f{Vector3f{bbox.min()[0], bbox.min()[1], bbox.min()[2]},
						Vector3f{bbox.max()[0], bbox.max()[1], bbox.max()[2]}};
	}

	AABB3f getBounds() const { return bounds; }

	float getDensity(const Vector3f& p) const {
		nanovdb::Vec3<float> pIndex =
			densityGrid->worldToIndexF(nanovdb::Vec3<float>(p.x(), p.y(), p.z()));
		return VDBSampler(densityGrid->tree())(pIndex);
	}

	float getMaxDensity() const { return maxDensity; }

private:
	AABB3f bounds;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> densityHandle;
	nanovdb::FloatGrid *densityGrid{nullptr};
	float maxDensity{0};
};

class HomogeneousMedium {
public:
	HomogeneousMedium() = default;

	HomogeneousMedium(Color sigma_a, Color sigma_s, Color L_e, float g) :
		sigma_a(sigma_a), sigma_s(sigma_s), L_e(L_e), phase(g) {}

	KRR_CALLABLE bool isEmissive() const { return !L_e.isZero(); }

	KRR_CALLABLE Color Le(Vector3f p) const { return L_e; }
	
	KRR_CALLABLE MediumProperties samplePoint(Vector3f p) const {
		return {sigma_a, sigma_s, &phase, L_e};
	}

	KRR_CALLABLE RayMajorant sampleRay(const Ray &ray, float tMax) const {
		return {sigma_a + sigma_s, 0, tMax};
	}

	Color sigma_a, sigma_s, L_e;
	HGPhaseFunction phase;
};

class NanoVDBMedium {
public:
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;

	NanoVDBMedium(const Affine3f& transform, Color sigma_a, Color sigma_s, float g, NanoVDBGrid density) :
		transform(transform), phase(g), sigma_a(sigma_a), sigma_s(sigma_s), densityGrid(std::move(density)) {
		inverseTransform				   = transform.inverse();
	}

	KRR_CALLABLE bool isEmissive() const { return false; }

	KRR_CALLABLE Color Le(Vector3f p) const { return 0; }

	KRR_CALLABLE MediumProperties samplePoint(Vector3f p) const { 
		p = inverseTransform * p;
		float d = densityGrid.getDensity(p);
		return {sigma_a * d, sigma_s * d, &phase, Le(p)};
	}

	KRR_CALLABLE RayMajorant sampleRay(const Ray &ray, float raytMax) const {
		// [TODO] currently we use a coarse majorant for the whole volume
		// but it seems that nanovdb has a built-in hierachical DDA on gpu?
		float tMin, tMax;
		Ray r = inverseTransform * ray;
		if (!densityGrid.getBounds().intersect(r.origin, r.dir, raytMax, &tMin, &tMax)) return {};
		return {densityGrid.getMaxDensity() * (sigma_a + sigma_s), tMin, tMax};
	}

	NanoVDBGrid densityGrid;
	Affine3f transform, inverseTransform;
	HGPhaseFunction phase;
	Color sigma_a, sigma_s;
};

/* Put these definitions here since the optix kernel will need them... */
/* Definitions of inline functions should be put into header files. */
KRR_CALLABLE bool Medium::isEmissive() const {
	auto emissive = [&](auto ptr) -> bool { return ptr->isEmissive(); };
	return dispatch(emissive);
}

KRR_CALLABLE MediumProperties Medium::samplePoint(Vector3f p) const {
	auto sample = [&](auto ptr) -> MediumProperties { return ptr->samplePoint(p); };
	return dispatch(sample);
}

KRR_CALLABLE RayMajorant Medium::sampleRay(const Ray &ray, float tMax) const {
	auto sample = [&](auto ptr) -> RayMajorant { return ptr->sampleRay(ray, tMax); };
	return dispatch(sample);
}


KRR_NAMESPACE_END