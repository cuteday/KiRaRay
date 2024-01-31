#include "py.h"
#include "common.h"

NAMESPACE_BEGIN(krr)

PYBIND11_MODULE(pykrr_common, m) {
	// used to find necessary dlls...
	m.attr("vulkan_root")  = KRR_VULKAN_ROOT;
	m.attr("pytorch_root") = KRR_PYTORCH_ROOT;
}

NAMESPACE_END(krr)