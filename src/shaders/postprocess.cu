#include "postprocess.h"
#include "math/utils.h"

KRR_NAMESPACE_BEGIN

namespace shader {
	using namespace math;

	constexpr uint n_linear_threads = 128;
	template <typename T>
	constexpr uint n_linear_blocks(T n_elements) { return (uint)divRoundUp(n_elements, (T)n_linear_threads); }
	template <typename K, typename T, typename ... Types>
	inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
		if (n_elements <= 0) {
			return;
		}
		kernel << <n_linear_blocks(n_elements), n_linear_threads, shmem_size, stream >> > ((uint)n_elements, args...);
	}

	template<typename T=float>
    __global__ void accumulateFrame(uint n_elements, T* currentBuffer, T* accumBuffer, uint accumCount)
    {
		uint i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_elements) return;
		float currentWeight = 1.f / (accumCount + 1);
		if (accumCount > 0) {
			// mocing average mode
			accumBuffer[i] = utils::lerp(accumBuffer[i], currentBuffer[i], currentWeight);
			// sum mode
			//accumBuffer[i] = (accumBuffer[i] + currentBuffer[i]) * currentWeight;
		}
		else {
			accumBuffer[i] = currentBuffer[i];
		}
	}
}

using namespace shader;

void AccumulatePass::render(CUDABuffer& frame, cudaStream_t stream) {

	linear_kernel(accumulateFrame<vec4f>, 0, stream, mFrameSize.x * mFrameSize.y, 
		frame.data<vec4f>(), mAccumBuffer.data<vec4f>(), mAccumCount);

	mAccumCount = min(mAccumCount + 1, mMaxAccumCount - 1);
}

KRR_NAMESPACE_END