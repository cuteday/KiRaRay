#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"
#include "math/math.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class AccumulatePass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<AccumulatePass>;

	AccumulatePass() = default;

	void renderUI() override {
		if (ui::CollapsingHeader("Accumulate pass")) {
			if (ui::Checkbox("Enabled", &mEnable)) reset();
			if (mEnable) {
				ui::Text("Accumulate count: %d\n", mAccumCount);
				if (ui::Button("reset")) reset();
			}
		}
	}

	void reset() { mAccumCount = 0; }

	void resize(const vec2i& size) override {
		mFrameSize = size;
		if (!mAccumBuffer) mAccumBuffer = new CUDABuffer();
		mAccumBuffer->resize(size.x * size.y * sizeof(vec4f));
		reset();
	}

	void render(CUDABuffer& frame);
	CUDABuffer& result() { return *mAccumBuffer; }

private:
	bool mEnable{ true };
	uint mAccumCount{ 0 };
	bool mMovingAverage{ false };
	uint mMaxAccumCount{ (uint)1e9 };
	CUDABuffer *mAccumBuffer;
};

KRR_NAMESPACE_END