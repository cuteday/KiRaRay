#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "shared.h"

namespace krr {
  
  // upload updated parameters each rt launch
  extern "C" __constant__ LaunchParams optixLaunchParams;

  extern "C" __global__ void KRR_RT_CH(radiance)()
  {
	  vec3f& prd = *getPRD<vec3f>();
	  int prim_id = optixGetPrimitiveIndex();
	  prd = vec3f(0.5);

  }
  
  extern "C" __global__ void KRR_RT_AH(radiance)()
  { 

  }

  extern "C" __global__ void KRR_RT_MS(radiance)()
  { 
	  *getPRD<vec3f>() = vec3f(1.0);
  }

  extern "C" __global__ void KRR_RT_RG(renderFrame)()
  {
    uint3 pixelID = optixGetLaunchIndex();
    const int frameID = optixLaunchParams.frameID;
    const uint32_t fbIndex = pixelID.x + pixelID.y * optixLaunchParams.fbSize.x;
	
	optixLaunchParams.colorBuffer[fbIndex] = vec4f(0.5f, 0.5f, 0.6f, 1.0f);

  }
  
} 
