#pragma once

#include <cuda.h>
#include <map>
#include <typeinfo>
#include <typeindex>

#include "common.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

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
void GPUCall(F &&func);

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
void GPUCall(F &&func) {
	GPUParallelFor(1, [=] KRR_DEVICE(int) mutable { func(); });
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

template <typename T> 
class DeviceAtomicCounter {
public:
	DeviceAtomicCounter() = default;

	KRR_CALLABLE void reset() {
#ifdef KRR_DEVICE_CODE
		mVal = 0;
#else
		cudaDeviceSynchronize();
		mVal.store(0, std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE int increment() {
#ifdef KRR_DEVICE_CODE
		return atomicAdd(&mVal, 1);
#else
		cudaDeviceSynchronize();
		return mVal.fetch_add(1, std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE int load() const {
#ifdef KRR_DEVICE_CODE
		return mVal;
#else
		cudaDeviceSynchronize();
		return mVal.load(std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE DeviceAtomicCounter &operator=(const DeviceAtomicCounter &w) {
#if defined(KRR_DEVICE_CODE)
		mVal = w.mVal;
#else
		cudaDeviceSynchronize();
		mVal.store(w.mVal.load());
#endif
		return *this;
	}

private:
#ifdef KRR_DEVICE_CODE
	int mVal{ 0 };
#else
	std::atomic<int> mVal{ 0 };
#endif
};


KRR_NAMESPACE_END
