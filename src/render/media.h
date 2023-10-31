#pragma once
#include "common.h"
#include "device/taggedptr.h"
#include "render/phase.h"
#include "medium.h"
#include "raytracing.h"
#include "util/volume.h"
#include "device/container.h"
#include "device/gpustd.h"

KRR_NAMESPACE_BEGIN

struct MajorantSegment {
	float tMin, tMax;
	SampledSpectrum sigma_maj;
};

class MajorantGrid : public Grid<float, 3> {
public:
	MajorantGrid() = default;

	MajorantGrid(AABB3f bounds, Vector3i res): 
		bounds(bounds) { this->res = res; }

	KRR_CALLABLE float lookup(int x, int y, int z) const {
		return voxels[x + res.x() * (y + res.y() * z)];
	}

	KRR_CALLABLE void set(int x, int y, int z, float v) {
		voxels[x + res.x() * (y + res.y() * z)] = v;
	}

	AABB3f bounds;
};

class MajorantIterator {
public:
	MajorantIterator() = default;

	KRR_CALLABLE MajorantIterator(Ray ray, float tMin, float tMax, Color sigma_t, const MajorantGrid *grid = nullptr) 
		: tMin(tMin), tMax(tMax), sigma_t(sigma_t), grid(grid) {
		if (!grid) return;	// acts like a homogeneous ray majorant iterator
		Ray rayGrid(grid->bounds.offset(ray.origin), ray.dir / grid->bounds.diagonal());
		Point3f gridIntersect = rayGrid(tMin);
		for (int axis = 0; axis < 3; ++axis) {
			// Initialize ray stepping parameters for _axis_
			// Compute current voxel for axis and handle negative zero direction
			voxel[axis]	 = clamp(gridIntersect[axis] * grid->res[axis], 0.f, grid->res[axis] - 1.f);
			deltaT[axis] = 1.f / (fabs(rayGrid.dir[axis]) * grid->res[axis]);
			if (rayGrid.dir[axis] == -0.f) rayGrid.dir[axis] = 0.f;
			if (rayGrid.dir[axis] >= 0) {
				// Handle ray with positive direction for voxel stepping
				float nextVoxelPos	= float(voxel[axis] + 1) / grid->res[axis];
				nextCrossingT[axis] = tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.dir[axis];
				step[axis]			= 1;
				voxelLimit[axis]	= grid->res[axis];
			} else {
				// Handle ray with negative direction for voxel stepping
				float nextVoxelPos	= float(voxel[axis]) / grid->res[axis];
				nextCrossingT[axis] = tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.dir[axis];
				step[axis]			= -1;
				voxelLimit[axis]	= -1;
			}
		}
	}

	KRR_CALLABLE gpu::optional<MajorantSegment> next() {
		if (tMin >= tMax) return {};
		if (!grid) {
			MajorantSegment seg{tMin, tMax, sigma_t};
			tMin = tMax;
			return seg;
		}
		int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
				   ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
				   ((nextCrossingT[1] < nextCrossingT[2]));
		const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
		int stepAxis		   = cmpToAxis[bits];
		float tVoxelExit	   = min(tMax, nextCrossingT[stepAxis]);

		// Get _maxDensity_ for current voxel and initialize _RayMajorantSegment_, _seg_
		Color sigma_maj = sigma_t * grid->lookup(voxel[0], voxel[1], voxel[2]);
		MajorantSegment seg{tMin, tVoxelExit, sigma_maj};

		// Advance to next voxel in maximum density grid
		tMin = tVoxelExit;
		if (nextCrossingT[stepAxis] > tMax) tMin = tMax;
		voxel[stepAxis] += step[stepAxis];
		if (voxel[stepAxis] == voxelLimit[stepAxis]) tMin = tMax;
		nextCrossingT[stepAxis] += deltaT[stepAxis];

		return seg;
	}

private:
	SampledSpectrum sigma_t;
	float tMin = M_FLOAT_INF, tMax = -M_FLOAT_INF;
	const MajorantGrid *grid = nullptr;
	Array3f nextCrossingT, deltaT;
	Array3i step, voxelLimit, voxel;
};

class HomogeneousMedium {
public:
	HomogeneousMedium() = default;

	HomogeneousMedium(RGB sigma_a, RGB sigma_s, RGB L_e, float g, 
		const RGBColorSpace* colorSpace = RGBColorSpace::sRGB) :
		sigma_a(sigma_a), sigma_s(sigma_s), L_e(L_e), phase(g), colorSpace(colorSpace) {}

	KRR_CALLABLE bool isEmissive() const { return L_e.any(); }

	KRR_CALLABLE SampledSpectrum Le(Vector3f p, const SampledWavelengths &lambda) const { 
#if KRR_RENDER_SPECTRAL
		return RGBUnboundedSpectrum(L_e, *colorSpace).sample(lambda); 
#else
		return L_e;
#endif
	}
	
	KRR_CALLABLE MediumProperties samplePoint(Vector3f p, const SampledWavelengths &lambda) const {
#if KRR_RENDER_SPECTRAL
		return {RGBUnboundedSpectrum(sigma_a, *colorSpace).sample(lambda),
				RGBUnboundedSpectrum(sigma_s, *colorSpace).sample(lambda), &phase, Le(p, lambda)};
#else 		
		return {sigma_a, sigma_s, &phase, L_e};
#endif
	}

	KRR_CALLABLE MajorantIterator sampleRay(const Ray &ray, float tMax,
											const SampledWavelengths &lambda) const {
#if KRR_RENDER_SPECTRAL
		return MajorantIterator{ray, 0, tMax,
								RGBUnboundedSpectrum(sigma_a, *colorSpace).sample(lambda) +
									RGBUnboundedSpectrum(sigma_s, *colorSpace).sample(lambda),
								nullptr};
#else
		return MajorantIterator{ray, 0, tMax, sigma_a + sigma_s, nullptr};
#endif
	}

	RGB sigma_a, sigma_s, L_e;
	HGPhaseFunction phase;
	const RGBColorSpace *colorSpace;
};

class NanoVDBMedium {
public:
	NanoVDBMedium(const Affine3f &transform, Color sigma_a, Color sigma_s, float g,
				  NanoVDBGrid density, const RGBColorSpace *colorSpace = RGBColorSpace::sRGB);

	KRR_HOST void initializeFromHost();

	KRR_CALLABLE bool isEmissive() const { return false; }

	KRR_CALLABLE SampledSpectrum Le(Vector3f p, const SampledWavelengths &lambda) const {
		return SampledSpectrum::Zero();
	}
	
	KRR_CALLABLE MediumProperties samplePoint(Vector3f p, const SampledWavelengths &lambda) const { 
		p = inverseTransform * p;
		float d = densityGrid.getDensity(p);
#if KRR_RENDER_SPECTRAL
		return {RGBUnboundedSpectrum(sigma_a, *colorSpace).sample(lambda) * d,
				RGBUnboundedSpectrum(sigma_s, *colorSpace).sample(lambda) * d, &phase,
				Le(p, lambda)};
#else
		return {sigma_a * d, sigma_s * d, &phase, Le(p, lambda)};
#endif
	}

	KRR_CALLABLE MajorantIterator sampleRay(const Ray &ray, float raytMax,
											const SampledWavelengths &lambda) const {
		float tMin = 0, tMax = raytMax;
		AABB3f box	 = densityGrid.getBounds();
		Ray localRay = inverseTransform * ray;
		if(!box.intersect(localRay.origin, localRay.dir, raytMax, &tMin, &tMax)) return {};
#if KRR_RENDER_SPECTRAL
		return MajorantIterator{localRay, tMin, tMax, 
			RGBUnboundedSpectrum(sigma_a, *colorSpace).sample(lambda) +
									RGBUnboundedSpectrum(sigma_s, *colorSpace).sample(lambda),
			&majorantGrid};
#else
		return MajorantIterator{localRay, tMin, tMax, sigma_a + sigma_s, &majorantGrid};
#endif
	}

	NanoVDBGrid densityGrid;
	MajorantGrid majorantGrid;
	Affine3f transform, inverseTransform;
	HGPhaseFunction phase;
	Color sigma_a, sigma_s;
	const RGBColorSpace *colorSpace;
};

/* Put these definitions here since the optix kernel will need them... */
/* Definitions of inline functions should be put into header files. */
inline SampledSpectrum Medium::Le(Vector3f p, const SampledWavelengths &lambda) const {
	auto Le = [&](auto ptr) -> SampledSpectrum { return ptr->Le(p, lambda); };
	return dispatch(Le);
}

inline bool Medium::isEmissive() const {
	auto emissive = [&](auto ptr) -> bool { return ptr->isEmissive(); };
	return dispatch(emissive);
}

inline MediumProperties Medium::samplePoint(Vector3f p, const SampledWavelengths &lambda) const {
	auto sample = [&](auto ptr) -> MediumProperties { return ptr->samplePoint(p, lambda); };
	return dispatch(sample);
}

inline MajorantIterator Medium::sampleRay(const Ray &ray, float tMax,
										  const SampledWavelengths &lambda) const {
	auto sample = [&](auto ptr) -> MajorantIterator { return ptr->sampleRay(ray, tMax, lambda); };
	return dispatch(sample);
}


KRR_NAMESPACE_END