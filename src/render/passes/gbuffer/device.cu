#include "device.h"
#include "render/shading.h"

KRR_NAMESPACE_BEGIN

extern "C" __constant__ LaunchParamsGBuffer launchParams;

extern "C" __global__ void KRR_RT_CH(Primary)() {

}

extern "C" __global__ void KRR_RT_MS(Primary)() {

}

extern "C" __global__ void KRR_RT_RG(Primary)() {

}

KRR_NAMESPACE_END