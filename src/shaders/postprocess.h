#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"
#include "math/math.h"
#include "gpu/buffer.h"

KRR_NAMESPACE_BEGIN

class AccumulatePass: public RenderPass {
public:
	using SharedPtr = std::shared_ptr<AccumulatePass>;

	AccumulatePass() = default;

	void renderUI() override {
		ui::Checkbox("Enabled", &mEnable);
		if (mEnable) {
			ui::Text("Accumulate count: %d\n", mAccumCount);
			if (ui::Button("reset")) {
				reset();
			}
		}
	}

	void reset() {
		mAccumCount = 0;
	}

	void resize(vec2i size) override{
		mFrameSize = size;
		mAccumBuffer.resize(size.x * size.y * sizeof(vec4f));
	}

	void render(CUDABuffer& frame, cudaStream_t stream = nullptr);
	CUDABuffer& result() { return mAccumBuffer; }

private:
	bool mEnable = true;
	uint mAccumCount = 0;
	uint mMaxAccumCount = 1e9;
	CUDABuffer mAccumBuffer;
};

class ToneMappingPass: public RenderPass {
public:
	using SharedPtr = std::shared_ptr<ToneMappingPass>;

	enum class Operator {
		Linear = 0,
		Reinhard,
		Aces,
		NumsOperators,
	};

	ToneMappingPass() = default;

	void renderUI() override;

	void setOperator(Operator toneMappingOperator)
		{mOperator = toneMappingOperator; }
	Operator getOperator() const { return mOperator; }

	void render(CUDABuffer& frame, cudaStream_t stream = nullptr);

private:
	bool mEnable = true;
	float mExposureCompensation = 1;
	Operator mOperator = Operator::Linear;
};

namespace shader {

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

	template<typename T>
	__global__ void accumulateFrame(uint n_elements, vec4f* currentBuffer, vec4f* accumBuffer, uint accumCount, bool average = false);

	template <typename T>
	__global__ void toneMap(uint n_elements, vec4f* frame, vec3f colorTransform, ToneMappingPass::Operator toneMapOperator);
}

KRR_NAMESPACE_END