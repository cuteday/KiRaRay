#pragma once
#ifndef __VKCUDA_H__
#define __VKCUDA_H__

#include <cuda_runtime_api.h>
#include "cuda.h"
#define CUDA_DRIVER_API
#include <helper_cuda.h>

bool isDeviceCompatible(void *Uuid, size_t size) {
  int cudaDevice = cudaInvalidDeviceId;
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp devProp = {};
    checkCudaErrors(cudaGetDeviceProperties(&devProp, i));
    if (!memcmp(&devProp.uuid, Uuid, size)) {
      cudaDevice = i;
      break;
    }
  }
  if (cudaDevice == cudaInvalidDeviceId) {
    return false;
  }

  int deviceSupportsHandle = 0;
  int attributeVal = 0;
  int deviceComputeMode = 0;

  checkCudaErrors(cuDeviceGetAttribute(
      &deviceComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cudaDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      cudaDevice));

#if defined(__linux__)
  checkCudaErrors(cuDeviceGetAttribute(
      &deviceSupportsHandle,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      cudaDevice));
#else
  checkCudaErrors(cuDeviceGetAttribute(
      &deviceSupportsHandle,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, cudaDevice));
#endif

  if ((deviceComputeMode != CU_COMPUTEMODE_DEFAULT) || !attributeVal ||
      !deviceSupportsHandle) {
    return false;
  }
  return true;
}

#endif  // __VKCUDA_H__
