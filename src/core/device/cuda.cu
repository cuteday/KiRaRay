
#include "device/cuda.h"

KRR_NAMESPACE_BEGIN

#ifdef __NVCC__

template <typename F>
__global__ void Kernel(F func, int nElements) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nElements) return;
	func(tid);
}

template <typename F>
void GPUParallelFor(int nElements, F func) {
	printf("Invoking GPU parallel for with %d threads BEFORE device code...\n", nElements);
	printf("Invoking GPU parallel for with %d threads ENTERED device code...\n", nElements);
	auto kernel = &Kernel<F>;
	int blockSize = GetBlockSize(kernel);
	int gridSize = (nElements + blockSize - 1) / blockSize;
	kernel << <gridSize, blockSize >> > (func, nElements);
#ifdef KRR_DEBUG_BUILD
	CUDA_SYNC_CHECK();
#endif
}

#endif

KRR_NAMESPACE_END