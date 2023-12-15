#pragma once
#include <filesystem>

#define NOMINMAX
#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#include "common.h"
#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

/* We did not use virtual device functions for VDB grids on purpose.
	For cuda device routines to access virtual functions, the class (i.e. virtual table)
	must also be created on device. Otherwise, each call to a virtual function will 
	attempt to access host memory (which causes the illegal memory access error). */

class NanoVDBGridBase {
public:
	using SharedPtr = std::shared_ptr<NanoVDBGridBase>;
	NanoVDBGridBase()  = default;
	~NanoVDBGridBase() = default;

	NanoVDBGridBase(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> &&density) :
		densityHandle(std::move(density)) {}

	KRR_HOST virtual void toDevice() { densityHandle.deviceUpload(); }

	KRR_CALLABLE AABB3f getBounds() const { return bounds; }

protected:
	AABB3f bounds;
	nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> densityHandle{};
};

template <typename DataType>
class NanoVDBGridBaseImpl: public NanoVDBGridBase {
public:
	NanoVDBGridBaseImpl() = default;
	~NanoVDBGridBaseImpl() = default;

	KRR_HOST NanoVDBGridBaseImpl(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> &&density,
		DataType maxDensity) : NanoVDBGridBase(std::move(density)), maxDensity(maxDensity) {}

	KRR_HOST NanoVDBGridBaseImpl(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> &&density) :
		NanoVDBGridBase(std::move(density)) { evalMaxDensity(); }

	KRR_CALLABLE DataType getMaxDensity() const { return maxDensity; }

protected:
	DataType maxDensity{0};
};

template <typename DataType> 
class NanoVDBGrid : public NanoVDBGridBaseImpl<DataType> {};

template<> class NanoVDBGrid<float> : public NanoVDBGridBaseImpl<float> {
public:
	using NativeType = float;
	using SharedPtr  = std::shared_ptr<NanoVDBGrid<float>>;
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;

	NanoVDBGrid() = default;

	NanoVDBGrid(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&& density, float maxDensity) :
		NanoVDBGridBaseImpl(std::move(density), maxDensity) {
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

	KRR_CALLABLE operator bool() const { return densityGrid != nullptr; }

	KRR_CALLABLE float getValue(const Vector3f &p) const {
		nanovdb::Vec3<float> pIndex =
			densityGrid->worldToIndexF(nanovdb::Vec3<float>(p.x(), p.y(), p.z()));
		return VDBSampler(densityGrid->tree())(pIndex);
	}

	KRR_CALLABLE nanovdb::FloatGrid *getNativeGrid() const { return densityGrid; }

protected:
	nanovdb::FloatGrid *densityGrid{nullptr};
};

template<> class NanoVDBGrid<Array3f> : public NanoVDBGridBaseImpl<Array3f> {
public:
	using NativeType = nanovdb::Vec3f;
	using SharedPtr  = std::shared_ptr<NanoVDBGrid<Array3f>>;
	using VDBSampler = nanovdb::SampleFromVoxels<nanovdb::Vec3fGrid::TreeType, 1, false>;

	NanoVDBGrid() = default;

	NanoVDBGrid(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> &&density, Array3f maxDensity) :
		NanoVDBGridBaseImpl(std::move(density), maxDensity) {
		densityGrid						   = densityHandle.grid<NativeType>();
		nanovdb::BBox<nanovdb::Vec3R> bbox = densityGrid->worldBBox();
		bounds = AABB3f{Vector3f{bbox.min()[0], bbox.min()[1], bbox.min()[2]},
						Vector3f{bbox.max()[0], bbox.max()[1], bbox.max()[2]}};
	}

	NanoVDBGrid(NanoVDBGrid&& other) noexcept {
		std::swap(densityHandle, other.densityHandle);
		densityGrid	  = densityHandle.grid<NativeType>();
		bounds		  = other.bounds;
		maxDensity	  = other.maxDensity;
	}

	KRR_CALLABLE operator bool() const { return densityGrid != nullptr; }

	KRR_CALLABLE Array3f getValue(const Vector3f &p) const {
		nanovdb::Vec3<float> pIndex =
			densityGrid->worldToIndexF(nanovdb::Vec3<float>(p.x(), p.y(), p.z()));
		auto color = VDBSampler(densityGrid->tree())(pIndex);
		return {color[0], color[1], color[2]};
	}

	KRR_CALLABLE nanovdb::Vec3fGrid *getNativeGrid() const { return densityGrid; }

protected:
	nanovdb::Vec3fGrid *densityGrid{nullptr};
};

NanoVDBGridBase::SharedPtr loadOpenVDB(std::filesystem::path path, std::string key);

KRR_NAMESPACE_END