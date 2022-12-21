#include "bdpt.h"
#include "render/shared.h"
#include "render/shading.h"
#include "util/hash.h"
#include "util/math_utils.h"

#include <optix_device.h>

using namespace krr;	// this is needed or nvcc cannot recognize the launchParams extern "C" var.

KRR_NAMESPACE_BEGIN

using namespace utils;
using namespace shader;
using namespace types;

extern "C" __global__ void KRR_RT_RG(GenerateCameraSubpath)() {
	uint workId(optixGetLaunchIndex().x);
}

extern "C" __global__ void KRR_RT_RG(GenerateLightSubpath)() {
	uint workId(optixGetLaunchIndex().x);
}

extern "C" __global__ void KRR_RT_RG(Connect)() {
	uint workId(optixGetLaunchIndex().x);
}

KRR_NAMESPACE_END