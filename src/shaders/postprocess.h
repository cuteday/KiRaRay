#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "math/math.h"
#include "gpu/buffer.h"

KRR_NAMESPACE_BEGIN

namespace shader {

	__global__ void accumulateFrame(uint n_elements, vec4f* currentBuffer, vec4f* accumBuffer, uint accumCount);

}

class AccumulatePass {
public:
	using SharedPtr = std::shared_ptr<AccumulatePass>;

	AccumulatePass() = default;

	void renderUI() {
		ui::Text("Accumulate count: %d\n", mAccumCount);
		if (ui::Button("reset")) {
			reset();
		}
	}

	void reset() {
		mAccumCount = 0;
	}

	void resize(const vec2i& size){
		mFrameSize = size;
		mAccumBuffer.resize(size.x * size.y * sizeof(vec4f));
	}

	void render(CUDABuffer& frame, cudaStream_t stream = nullptr);
	CUDABuffer& result() { return mAccumBuffer; }

private:
	vec2i mFrameSize;
	uint mAccumCount = 0;
	uint mMaxAccumCount = 1e9;
	CUDABuffer mAccumBuffer;
};



class ToneMappingPass {
public:
	ToneMappingPass() = default;

private:

};

KRR_NAMESPACE_END