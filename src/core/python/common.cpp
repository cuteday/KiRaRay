#include "py.h"
#include "common.h"

NAMESPACE_BEGIN(krr)

PYBIND11_MODULE(pykrr_common, m) {
	// used to find necessary dlls...
	m.attr("vulkan_root")  = KRR_VULKAN_ROOT;
	m.attr("pytorch_root") = KRR_PYTORCH_ROOT;
	m.attr("project_root") = KRR_PROJECT_DIR;
	m.attr("openvdb_io") = bool(KRR_ENABLE_OPENVDB_IO);
	m.attr("usd_root") = KRR_ENABLE_USD ? KRR_USD_ROOT : "";
}

NAMESPACE_END(krr)
