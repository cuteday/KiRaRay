#include <fstream>
#include "volume.h"

#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

#if defined (KRR_DEBUG_BUILD)
#pragma comment(lib, "openvdb_d.lib")
#pragma comment(lib, "tbb_debug.lib")
#else
#pragma comment(lib, "openvdb.lib")
#pragma comment(lib, "tbb.lib")
#endif

#include "logger.h"

KRR_NAMESPACE_BEGIN

NanoVDBGrid<float>::SharedPtr loadNanoVDB(std::filesystem::path path, string key) {
	openvdb::initialize();
	Log(Info, "Loading openvdb file from %s", path.string().c_str());
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

	auto grid							  = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
	auto transform						  = grid->transform();
	auto handle							  = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grid);
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();
	if (metadata->gridType() != nanovdb::GridType::Float) Log(Fatal, "only support float grid!");
	float minValue, maxValue;
	grid->evalMinMax(minValue, maxValue);
	return std::make_shared<NanoVDBGrid<float>>(std::move(handle), maxValue);
}

KRR_NAMESPACE_END