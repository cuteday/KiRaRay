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
	Spectrum sigma_maj;
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

template <typename GridType>
void initializeMajorantGrid(MajorantGrid &majorantGrid, GridType *floatGrid);

class MajorantIterator {
public:
	MajorantIterator() = default;

	KRR_CALLABLE MajorantIterator(Ray ray, float tMin, float tMax, Spectrum sigma_t, const MajorantGrid *grid = nullptr) 
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
		Spectrum sigma_maj = sigma_t * grid->lookup(voxel[0], voxel[1], voxel[2]);
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
	Spectrum sigma_t;	// used as sigma_majorant if grid is nullptr
	float tMin = M_FLOAT_INF, tMax = -M_FLOAT_INF;
	const MajorantGrid *grid = nullptr;
	Array3f nextCrossingT, deltaT;
	Array3i step, voxelLimit, voxel;
};

class HomogeneousMedium {
public:
	HomogeneousMedium() = default;

	HomogeneousMedium(RGB sigma_t, RGB albedo, RGB L_e, float g, 
		const RGBColorSpace* colorSpace = KRR_DEFAULT_COLORSPACE) :
		sigma_t(sigma_t), albedo(albedo), L_e(L_e), phase(g), colorSpace(colorSpace) {}

	KRR_CALLABLE bool isEmissive() const { return L_e.any(); }

	KRR_CALLABLE MediumProperties samplePoint(Vector3f p, const SampledWavelengths &lambda) const {
		Spectrum sigma_t_spec = Spectrum::fromRGB(sigma_t, SpectrumType::RGBUnbounded, lambda, *colorSpace);
		Spectrum sigma_s_spec =
			sigma_t_spec * Spectrum::fromRGB(albedo, SpectrumType::RGBUnbounded, lambda, *colorSpace);
		return {sigma_t_spec - sigma_s_spec, sigma_s_spec, &phase, Le(p, lambda)};
	}

	KRR_CALLABLE MajorantIterator sampleRay(const Ray &ray, float tMax,
											const SampledWavelengths &lambda) const {
		return MajorantIterator{ ray, 0, tMax,
			Spectrum::fromRGB(sigma_t, SpectrumType::RGBUnbounded, lambda, *colorSpace), nullptr};
	}

	RGB sigma_t, albedo, L_e;
	HGPhaseFunction phase;
	const RGBColorSpace *colorSpace;

protected:
	KRR_CALLABLE Spectrum Le(Vector3f p, const SampledWavelengths &lambda) const {
		return Spectrum::fromRGB(L_e, SpectrumType::RGBIlluminant, lambda, *colorSpace);
	}
};

template <typename DataType>
class NanoVDBMedium {
public:
	NanoVDBMedium(const Affine3f &transform, RGB sigma_t, RGB albedo, float g, 
		NanoVDBGrid<DataType> density, NanoVDBGrid<float> temperature, NanoVDBGrid<Array3f> albedoGrid,
		float scale, float LeScale, float temperatureScale, float temperatureOffset,
		const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE);

	KRR_HOST void initializeFromHost();

	KRR_CALLABLE bool isEmissive() const { return false; }
	
	KRR_CALLABLE MediumProperties samplePoint(Vector3f p, const SampledWavelengths &lambda) const { 
		p = inverseTransform * p;
		Spectrum sigma_t_spec;
		if constexpr (std::is_same_v<DataType, float>)
			sigma_t_spec = densityGrid.getValue(p) * scale *
				Spectrum::fromRGB(sigma_t, SpectrumType::RGBUnbounded, lambda, *colorSpace);
		else sigma_t_spec = Spectrum::fromRGB(densityGrid.getValue(p), SpectrumType::RGBUnbounded,
										lambda, *colorSpace) * scale;
		Spectrum sigma_s_spec = sigma_t_spec * Spectrum::fromRGB(
			albedoGrid ? albedoGrid.getValue(p) : albedo, SpectrumType::RGBUnbounded, lambda, *colorSpace);
		return {sigma_t_spec - sigma_s_spec, sigma_s_spec, &phase, Le(p, lambda)};
	}

	KRR_CALLABLE MajorantIterator sampleRay(const Ray &ray, float raytMax,
											const SampledWavelengths &lambda) const {
		float tMin = 0, tMax = raytMax;
		AABB3f box	 = densityGrid.getBounds();
		Ray localRay = inverseTransform * ray;
		if(!box.intersect(localRay.origin, localRay.dir, raytMax, &tMin, &tMax)) return {};
		if (std::is_same_v<DataType, float>)
			return MajorantIterator{localRay, tMin, tMax, scale * 
			Spectrum::fromRGB(sigma_t, SpectrumType::RGBUnbounded, lambda, *colorSpace), &majorantGrid};
		else 
			return MajorantIterator{localRay, tMin, tMax, Spectrum::Constant(scale), &majorantGrid};
	}

	NanoVDBGrid<DataType> densityGrid;
	NanoVDBGrid<float> temperatureGrid;
	NanoVDBGrid<Array3f> albedoGrid;
	MajorantGrid majorantGrid;
	Affine3f transform, inverseTransform;
	HGPhaseFunction phase;
	RGB sigma_t, albedo;
	float temperatureScale, temperatureOffset;
	float LeScale, scale;
	const RGBColorSpace *colorSpace;

protected:
	/* Le() should only be called by sampledPoint, and its argument p is in local coords. */
	KRR_CALLABLE Spectrum Le(Vector3f p, const SampledWavelengths &lambda) const {
		if (!temperatureGrid) return Spectrum::Zero();
		float temp = (temperatureGrid.getValue(p) - temperatureOffset) * temperatureScale;
#if KRR_RENDER_SPECTRAL
		return LeScale * BlackbodySpectrum(temp).sample(lambda);
#else 
		return LeScale * BlackbodySpectrum(temp).sample(lambda).toRGB(lambda, *colorSpace);
#endif
	}
};

template <typename DataType>
NanoVDBMedium<DataType>::NanoVDBMedium(const Affine3f &transform, RGB sigma_t, RGB albedo, float g,
									   NanoVDBGrid<DataType> density,
									   NanoVDBGrid<float> temperature,
									   NanoVDBGrid<Array3f> albedoGrid, float scale, float LeScale,
									   float temperatureScale, float temperatureOffset,
									   const RGBColorSpace *colorSpace) :
	transform(transform), phase(g), sigma_t(sigma_t), albedo(albedo), densityGrid(std::move(density)), 
	temperatureGrid(std::move(temperature)), albedoGrid(std::move(albedoGrid)), 
	scale(scale), LeScale(LeScale), temperatureScale(temperatureScale), 
	temperatureOffset(temperatureOffset), colorSpace(colorSpace) {
	inverseTransform = transform.inverse();
	const Vector3f majorantGridRes{64, 64, 64};
	majorantGrid	 = MajorantGrid(densityGrid.getBounds(), majorantGridRes);
	if (albedoGrid) albedo = 1;	// albedo is deprecated if albedoGrid is provided
	if constexpr (std::is_same_v<DataType, Array3f>) sigma_t = 1;	// sigma_t is deprecated if densityGrid is RGB

}

template <typename DataType> 
void NanoVDBMedium<DataType>::initializeFromHost() {
	densityGrid.toDevice();
	initializeMajorantGrid(majorantGrid, densityGrid.getNativeGrid());
}

/* Put these definitions here since the optix kernel will need them... */
/* Definitions of inline functions should be put into header files. */

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