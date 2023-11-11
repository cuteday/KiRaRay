#include "media.h"

#include "medium.h"
#include "raytracing.h"
#include "device/cuda.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN
/* Note that the function qualifier (e.g. inline) should be consistent between declaration and definition. */

void initializeMajorantGrid(MajorantGrid& majorantGrid,
							nanovdb::FloatGrid *floatGrid) {
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
			nanovdb::Vec3R i0 = floatGrid->worldToIndexF(
				nanovdb::Vec3R(wb.min().x(), wb.min().y(), wb.min().z()));
			nanovdb::Vec3R i1 = floatGrid->worldToIndexF(
				nanovdb::Vec3R(wb.max().x(), wb.max().y(), wb.max().z()));

			// Now find integer index-space bounds, accounting for both
			// filtering and the overall index bounding box.
			auto bbox	= floatGrid->indexBBox();
			float delta = 1.f; // Filter slop
			int nx0		= max(int(i0[0] - delta), bbox.min()[0]);
			int nx1		= min(int(i1[0] + delta), bbox.max()[0]);
			int ny0		= max(int(i0[1] - delta), bbox.min()[1]);
			int ny1		= min(int(i1[1] + delta), bbox.max()[1]);
			int nz0		= max(int(i0[2] - delta), bbox.min()[2]);
			int nz1		= min(int(i1[2] + delta), bbox.max()[2]);

			float maxValue = 0;
			auto accessor  = floatGrid->getAccessor();

			for (int nz = nz0; nz <= nz1; ++nz)
				for (int ny = ny0; ny <= ny1; ++ny)
					for (int nx = nx0; nx <= nx1; ++nx)
						maxValue = max(maxValue, accessor.getValue({nx, ny, nz}));

			majorantGrid.set(x, y, z, maxValue);
		}, 0);
}

NanoVDBMedium::NanoVDBMedium(const Affine3f &transform, RGB sigma_a, RGB sigma_s, float g,
							 NanoVDBGrid density, NanoVDBGrid temperature, const RGBColorSpace *colorSpace) :
	transform(transform), phase(g), sigma_a(sigma_a), sigma_s(sigma_s), densityGrid(std::move(density)), 
	temperatureGrid(std::move(temperature)), colorSpace(colorSpace) {
	inverseTransform = transform.inverse();
	const Vector3f majorantGridRes{64, 64, 64};
	majorantGrid	 = MajorantGrid(densityGrid.getBounds(), majorantGridRes);
}

void NanoVDBMedium::initializeFromHost() {
	densityGrid.toDevice();
	initializeMajorantGrid(majorantGrid, densityGrid.getFloatGrid());
}

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