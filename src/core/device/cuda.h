#pragma once

#include <map>
#include <typeinfo>
#include <typeindex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "render/color.h"
#include "device/taggedptr.h"

KRR_NAMESPACE_BEGIN

class CudaRenderTarget {
public:
	CudaRenderTarget() = default;
	KRR_CALLABLE CudaRenderTarget(cudaSurfaceObject_t cudaFrame, int width,
								  int height) :
		mCudaFrame(cudaFrame), width(width), height(height) {}
	~CudaRenderTarget() = default;

	KRR_DEVICE RGBA read(int x, int y) const {
		float4 res{};
#ifdef __NVCC__
		surf2Dread(&res, mCudaFrame, x * sizeof(float4), height - 1 - y);
		return RGBA(res);
#endif
		return {};
	}
	KRR_DEVICE void write(const RGBA &value, int x, int y) {
#ifdef __NVCC__
		surf2Dwrite(float4(value), mCudaFrame, x * sizeof(float4), height - 1 - y);
#endif
	}
	KRR_DEVICE RGBA read(int idx) const {
		int x = idx % width, y = idx / width;
		return read(x, y);
	}
	KRR_DEVICE void write(const RGBA &value, int idx) {
		int x = idx % width, y = idx / width;
		return write(value, x, y);
	}

	KRR_CALLABLE operator cudaSurfaceObject_t() const { return mCudaFrame; }
	KRR_CALLABLE operator bool() const { return mCudaFrame != 0; }
	KRR_CALLABLE bool isValid() const { return mCudaFrame != 0; }

	cudaSurfaceObject_t mCudaFrame{};
	int width, height;
};

template <typename F>
inline int GetBlockSize(F kernel) {
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

template <typename F>
void GPUParallelFor(int nElements, F func, CUstream stream = 0);

template <typename F> 
void GPUCall(F &&func, CUstream stream = 0);

template <typename K, typename T, typename... Types>
void LinearKernelShmem(K kernel, uint32_t shmemSize, cudaStream_t stream, T n_elements,
						  Types... args);

template <typename K, typename T, typename... Types>
void LinearKernel(K kernel, cudaStream_t stream, T n_elements, Types... args);


#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nElements) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) return;
	func(tid);
}

template <typename F>
inline void GPUParallelFor(int nElements, F func, CUstream stream) {
	auto kernel = &Kernel<F>;
	int blockSize = GetBlockSize(kernel);
	int gridSize = (nElements + blockSize - 1) / blockSize;
	kernel <<<gridSize, blockSize, 0, stream>>> (func, nElements);
#ifdef KRR_DEBUG_BUILD
	CUDA_SYNC_CHECK();
#endif
}

template <typename F> 
void GPUCall(F &&func, CUstream stream) {
	GPUParallelFor(1, [=] KRR_DEVICE(int) mutable { func(); }, stream);
}

template <typename K, typename T, typename... Types>
inline void LinearKernelShmem(K kernel, uint32_t shmemSize, cudaStream_t stream, T nElements,
						  Types... args) {
	if (nElements <= 0) return;
	int blockSize = GetBlockSize(kernel);
	int gridSize  = (nElements + blockSize - 1) / blockSize;
	kernel<<<gridSize, blockSize, shmemSize, stream>>>(
		(uint32_t) nElements, args...);
}

template <typename K, typename T, typename... Types>
inline void LinearKernel(K kernel, cudaStream_t stream, T n_elements, Types... args) {
	LinearKernelShmem(kernel, 0, stream, n_elements, std::forward<Types>(args)...);
}
#endif

KRR_NAMESPACE_END
