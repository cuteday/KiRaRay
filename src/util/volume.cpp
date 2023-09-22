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

NanoVDBGrid::SharedPtr loadNanoVDB(std::filesystem::path path) {
	openvdb::initialize();
	Log(Info, "Loading openvdb file from %s", path.string().c_str());
	openvdb::io::File file(path.generic_string());
	if (!file.open())
		Log(Fatal, "Failed to open vdb file %s", path.string().c_str());
	
	openvdb::GridBase::Ptr baseGrid;
	if(file.hasGrid("density"))
		baseGrid = file.readGrid("density");
	else {
		baseGrid = file.getGrids()->at(0);
		Log(Warning, "VDB file do not pocess a density grid, loading the first by default.");
	}

	auto grid							  = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
	auto transform						  = grid->transform();
	auto handle							  = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grid);
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();
	Log(Info, "Find a vdb grid: %s", metadata->gridName());
	if (metadata->gridType() != nanovdb::GridType::Float) Log(Fatal, "only support float grid!");
	float minValue, maxValue;
	grid->evalMinMax(minValue, maxValue);
	return std::make_shared<NanoVDBGrid>(std::move(handle), maxValue);
}

KRR_NAMESPACE_END