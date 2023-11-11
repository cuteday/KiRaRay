#pragma once
#include <filesystem>

#define NOMINMAX
#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

class NanoVDBGrid {
public:
	using SharedPtr  = std::shared_ptr<NanoVDBGrid>;
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;

	NanoVDBGrid() = default;

	NanoVDBGrid(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&& density, float maxDensity) :
		densityHandle(std::move(density)), maxDensity(maxDensity) {
		densityGrid						   = densityHandle.grid<float>();
		nanovdb::BBox<nanovdb::Vec3R> bbox = densityGrid->worldBBox();
		bounds = AABB3f{Vector3f{bbox.min()[0], bbox.min()[1], bbox.min()[2]},
						Vector3f{bbox.max()[0], bbox.max()[1], bbox.max()[2]}};
	}

	NanoVDBGrid(NanoVDBGrid&& other) noexcept {
		std::swap(densityHandle, other.densityHandle);
		densityGrid	  = densityHandle.grid<float>();
		bounds		  = other.bounds;
		maxDensity	  = other.maxDensity;
	}

	void toDevice() { densityHandle.deviceUpload(); }

	KRR_CALLABLE operator bool() const { return densityGrid != nullptr; }

	KRR_CALLABLE AABB3f getBounds() const { return bounds; }

	KRR_CALLABLE float getDensity(const Vector3f &p) const {
		nanovdb::Vec3<float> pIndex =
			densityGrid->worldToIndexF(nanovdb::Vec3<float>(p.x(), p.y(), p.z()));
		return VDBSampler(densityGrid->tree())(pIndex);
	}

	KRR_CALLABLE float getMaxDensity() const { return maxDensity; }

	KRR_CALLABLE nanovdb::FloatGrid *getFloatGrid() const { return densityGrid; }

private:
	AABB3f bounds;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> densityHandle{};
	nanovdb::FloatGrid *densityGrid{nullptr};
	float maxDensity{0};
};

NanoVDBGrid::SharedPtr loadNanoVDB(std::filesystem::path path, std::string key);

KRR_NAMESPACE_END