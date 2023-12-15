#include <fstream>
#include "volume.h"

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>

#if defined (KRR_DEBUG_BUILD)
#pragma comment(lib, "openvdb_d.lib")
#pragma comment(lib, "tbb_debug.lib")
#else
#pragma comment(lib, "openvdb.lib")
#pragma comment(lib, "tbb.lib")
#endif

#include "logger.h"

KRR_NAMESPACE_BEGIN

NanoVDBGridBase::SharedPtr loadNanoVDB(std::filesystem::path path) {
	Log(Info, "Loading nanovdb file from %s", path.string().c_str());
	
	auto handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(path.generic_string());
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();

	if (metadata->gridType() == nanovdb::GridType::Float) {
		float minValue, maxValue;
		auto grid = handle.grid<float>();
		grid->tree().extrema(minValue, maxValue);
		return std::make_shared<NanoVDBGrid<float>>(std::move(handle), maxValue);
	} else if (metadata->gridType() == nanovdb::GridType::Vec3f) {
		auto grid = handle.grid<nanovdb::Vec3f>();
		nanovdb::Vec3f minValue, maxValue;
		grid->tree().extrema(minValue, maxValue);
		return std::make_shared<NanoVDBGrid<Array3f>>(std::move(handle), 
						Array3f{maxValue[0], maxValue[1], maxValue[2]});
	} else {
		Log(Fatal, "Unsupported data type for nanovdb grid!");
		return nullptr;
	}

	return nullptr;
}

NanoVDBGridBase::SharedPtr loadOpenVDB(std::filesystem::path path, string key) {
	openvdb::initialize();
	Log(Info, "Loading openvdb grid %s from file %s", key.c_str(), path.string().c_str());
	openvdb::io::File file(path.generic_string());
	if (!file.open())
		Log(Fatal, "Failed to open vdb file %s", path.string().c_str());
	
	openvdb::GridBase::Ptr baseGrid;
	if (file.hasGrid(key.c_str()))
		baseGrid = file.readGrid(key.c_str());
	else {
		Log(Warning, "VDB file %s do not pocess a [%s] grid.", path.string().c_str(), key.c_str());
		return nullptr;
	}

	auto handle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(baseGrid);
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();
	if (metadata->gridType() == nanovdb::GridType::Float) {
		float minValue, maxValue;
		auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
		grid->evalMinMax(minValue, maxValue);
		return std::make_shared<NanoVDBGrid<float>>(std::move(handle), maxValue);
	} else if (metadata->gridType() == nanovdb::GridType::Vec3f) {
		auto grid = openvdb::gridPtrCast<openvdb::Vec3fGrid>(baseGrid);
		openvdb::Vec3f minValue, maxValue;
		grid->evalMinMax(minValue, maxValue);
		return std::make_shared<NanoVDBGrid<Array3f>>(std::move(handle), 
			Array3f{maxValue[0], maxValue[1], maxValue[2]});
	} else {
		Log(Fatal, "Unsupported data type for openvdb grid!");
		return nullptr;
	}
}

KRR_NAMESPACE_END