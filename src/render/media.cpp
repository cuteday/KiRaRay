#include "media.h"

#include "medium.h"
#include "raytracing.h"
#include "device/cuda.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN
/* Note that the function qualifier (e.g. inline) should be consistent between declaration and definition. */

template <typename GridType>
void initializeMajorantGrid(MajorantGrid &majorantGrid, GridType *densityGrid) {
	auto res = majorantGrid.res;
	cudaDeviceSynchronize();
	// [TODO] This device memory is not properly freed
	majorantGrid.voxels = TypedBuffer<float>(res.x() * res.y() * res.z());
	GPUParallelFor(res.x() * res.y() * res.z(),
		[=] KRR_DEVICE(int index) mutable {
			int x = index % majorantGrid.res.x();
			int y = (index / majorantGrid.res.x()) % majorantGrid.res.y();
			int z = index / (majorantGrid.res.x() * majorantGrid.res.y());
			DCHECK_EQ(index, x + majorantGrid.res.x() * (y + majorantGrid.res.y() * z));

			// World (aka medium) space bounds of this max grid cell
			AABB3f wb(majorantGrid.bounds.lerp(Vector3f(float(x) / majorantGrid.res.x(),
														float(y) / majorantGrid.res.y(),
														float(z) / majorantGrid.res.z())),
					  majorantGrid.bounds.lerp(Vector3f(float(x + 1) / majorantGrid.res.x(),
														float(y + 1) / majorantGrid.res.y(),
														float(z + 1) / majorantGrid.res.z())));

			// Compute corresponding NanoVDB index-space bounds in floating-point.
			nanovdb::Vec3R i0 = densityGrid->worldToIndexF(
				nanovdb::Vec3R(wb.min().x(), wb.min().y(), wb.min().z()));
			nanovdb::Vec3R i1 = densityGrid->worldToIndexF(
				nanovdb::Vec3R(wb.max().x(), wb.max().y(), wb.max().z()));

			// Now find integer index-space bounds, accounting for both
			// filtering and the overall index bounding box.
			auto bbox	= densityGrid->indexBBox();
			float delta = 1.f; // Filter slop
			int nx0		= max(int(i0[0] - delta), bbox.min()[0]);
			int nx1		= min(int(i1[0] + delta), bbox.max()[0]);
			int ny0		= max(int(i0[1] - delta), bbox.min()[1]);
			int ny1		= min(int(i1[1] + delta), bbox.max()[1]);
			int nz0		= max(int(i0[2] - delta), bbox.min()[2]);
			int nz1		= min(int(i1[2] + delta), bbox.max()[2]);

			float maxValue = 0;
			auto accessor  = densityGrid->getAccessor();

			for (int nz = nz0; nz <= nz1; ++nz)
				for (int ny = ny0; ny <= ny1; ++ny)
					for (int nx = nx0; nx <= nx1; ++nx)
						if constexpr (std::is_same_v<GridType, nanovdb::FloatGrid>) {
							maxValue = max(maxValue, accessor.getValue({nx, ny, nz}));
						} else if constexpr (std::is_same_v<GridType, nanovdb::Vec3fGrid>) {
							auto value = accessor.getValue({nx, ny, nz});
							RGB color  = {value[0], value[1], value[2]};
							RGBUnboundedSpectrum spectrum(color, *KRR_DEFAULT_COLORSPACE_GPU);
							maxValue = max(maxValue, spectrum.maxValue());
						} else {
							static_assert(false, "Unsupported grid type!");
						}
			majorantGrid.set(x, y, z, maxValue);
		}, 0);
}

// explicit instantiation of enable data types of vdb grids
template void initializeMajorantGrid<nanovdb::FloatGrid>(MajorantGrid &majorantGrid,
														 nanovdb::FloatGrid *densityGrid);
template void initializeMajorantGrid<nanovdb::Vec3fGrid>(MajorantGrid &majorantGrid,
														 nanovdb::Vec3fGrid *densityGrid);

KRR_HOST_DEVICE PhaseFunctionSample HGPhaseFunction::sample(const Vector3f &wo,
														 const Vector2f &u) const {
	float g = clamp(this->g, -.99f, .99f);

	// Compute $\cos\theta$ for Henyey-Greenstein sample
	float cosTheta;
	if (fabs(g) < 1e-3f)
		cosTheta = 1 - 2 * u[0];
	else
		cosTheta = -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * u[0])));

	// Compute direction _wi_ for Henyey-Greenstein sample
	float sinTheta = safe_sqrt(1 - pow2(cosTheta));
	float phi	   = M_2PI * u[1];
	Vector3f wi = Frame(wo.normalized()).toWorld(utils::sphericalToCartesian(sinTheta, cosTheta, phi));
	
	float pdf = this->pdf(wo.normalized(), wi);
	return PhaseFunctionSample{wi, pdf, pdf};
}

KRR_HOST_DEVICE float HGPhaseFunction::pdf(const Vector3f &wo, const Vector3f &wi) const {
	return p(wo, wi);
}

KRR_HOST_DEVICE float HGPhaseFunction::p(const Vector3f &wo, const Vector3f &wi) const {
	float g			= clamp(this->g, -.99f, .99f);
	float denom = 1 + pow2(g) + 2 * g * wo.dot(wi);
	return M_INV_4PI * (1 - pow2(g)) / (denom * safe_sqrt(denom));
}

KRR_HOST_DEVICE PhaseFunctionSample PhaseFunction::sample(const Vector3f &wo, const Vector2f &u) const {
	auto sample = [&](auto ptr) -> PhaseFunctionSample { return ptr->sample(wo, u); };
	return dispatch(sample);
}

KRR_HOST_DEVICE float PhaseFunction::pdf(const Vector3f &wo, const Vector3f &wi) const {
	auto pdf = [&](auto ptr) -> float { return ptr->pdf(wo, wi); };
	return dispatch(pdf);
}

KRR_HOST_DEVICE float PhaseFunction::p(const Vector3f &wo, const Vector3f &wi) const {
	auto p = [&](auto ptr) -> float { return ptr->p(wo, wi); };
	return dispatch(p);
}


KRR_NAMESPACE_END