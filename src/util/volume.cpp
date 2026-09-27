#include "volume.h"
#include "logger.h"

NAMESPACE_BEGIN(krr)

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

NAMESPACE_END(krr)
