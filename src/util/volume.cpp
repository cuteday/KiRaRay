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

using namespace krr;

nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> loadNanoVDB(std::filesystem::path path, float* maxDensity) {
	std::ifstream is(path, std::ios_base::binary);
	if (!is.good()) Log(Fatal, "failed to open file");
	
	auto grids = openvdb::io::Stream(is).getGrids();
	openvdb::MetaMap::Ptr metaData = openvdb::io::Stream(is).getMetadata();
	auto grid  = openvdb::GridBase::grid<openvdb::FloatGrid>(grids->at(0));
	auto handle							  = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grid);
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();
	Log(Info, "Find a vdb grid: %s", metadata->gridName());
	if (metadata->gridType() != nanovdb::GridType::Float) Log(Fatal, "only support float grid!");
	
	float minValue, maxValue;
	grid->evalMinMax(minValue, maxValue);
	if (maxDensity != nullptr) *maxDensity = maxValue;
	return handle;
}
