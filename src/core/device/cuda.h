#pragma once

#include <cuda.h>
#include <map>
#include <typeinfo>
#include <typeindex>

#include "device/optix.h"
#include "common.h"
#include "taggedptr.h"

#define CUDA_CHECK(call)							                    \
	{									                                \
	  cudaError_t rc = call;                                            \
	  if (rc != cudaSuccess) {                                          \
		std::stringstream ss;                                           \
		cudaError_t err =  rc; /*cudaGetLastError();*/                  \
		ss << "CUDA Error " << cudaGetErrorName(err)                    \
			<< " (" << cudaGetErrorString(err) << ")";                  \
		logError(ss.str());                                             \
		throw std::runtime_error(ss.str());                             \
	  }                                                                 \
	}


#define CUDA_SYNC_CHECK()                                               \
  {																		\
	cudaDeviceSynchronize();                                            \
	cudaError_t error = cudaGetLastError();                             \
	if( error != cudaSuccess )                                          \
	  {                                                                 \
		fprintf( stderr, "error (%s: line %d): %s\n",                   \
			__FILE__, __LINE__, cudaGetErrorString( error ) );          \
		throw std::runtime_error("CUDA synchronized check failed");                                                           \
	  }                                                                 \
  }


KRR_NAMESPACE_BEGIN

template <typename F>
inline int GetBlockSize(F kernel) {
	// How to configure block size: 
	// https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
	static std::map<std::type_index, int> kernelBlockSizes;

	std::type_index index = std::type_index(typeid(F));

	auto iter = kernelBlockSizes.find(index);
	if (iter != kernelBlockSizes.end())
		return iter->second;

	int minGridSize, blockSize;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
	kernelBlockSizes[index] = blockSize;
	return blockSize;
}

#ifdef __NVCC__

template <typename F>
__global__ void Kernel(F func, int nElements) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) return;
	func(tid);
}

template <typename F>
void GPUParrallelFor(int nElements, F func) {
	auto kernel = &Kernel<F>;
	int blockSize = GetBlockSize(kernel);
	int gridSize = (nElements + blockSize - 1) / blockSize;
	kernel <<<gridSize, blockSize >>> (func, nElements);
#ifdef KRR_DEBUG_BUILD
	CUDA_SYNC_CHECK();
#endif
}

#endif

KRR_NAMESPACE_END
