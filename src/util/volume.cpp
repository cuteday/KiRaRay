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
	std::ifstream is(path, std::ios_base::binary);
	if (!is.good()) Log(Fatal, "failed to open file");
	Log(Info, "Loading openvdb file from %s", path.string().c_str());
	openvdb::GridPtrVecPtr grids = openvdb::io::Stream(is).getGrids();
	openvdb::MetaMap::Ptr metaData = openvdb::io::Stream(is).getMetadata();
	Log(Info, "Loading VDB data with %zd grids...", grids->size());
	auto grid  = openvdb::GridBase::grid<openvdb::FloatGrid>(grids->at(0));
	auto transform = grid->transform();

	auto handle							  = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grid);
	const nanovdb::GridMetaData *metadata = handle.gridMetaData();
	Log(Info, "Find a vdb grid: %s", metadata->gridName());
	if (metadata->gridType() != nanovdb::GridType::Float) Log(Fatal, "only support float grid!");
	float minValue, maxValue;
	grid->evalMinMax(minValue, maxValue);
	return std::make_shared<NanoVDBGrid>(std::move(handle), maxValue);
}

KRR_NAMESPACE_END