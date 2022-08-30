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
	KRR_REGISTER_PASS_DEC(AccumulatePass);
	KRR_CLASS_DEFINE(AccumulatePass, mMovingAverage);

	AccumulatePass() = default;

	void renderUI() override;

	void reset() { mAccumCount = 0; }

	void resize(const Vector2i& size) override {
		mFrameSize = size;
		if (!mAccumBuffer) mAccumBuffer = new CUDABuffer();
		mAccumBuffer->resize(size[0] * size[1] * sizeof(Vector4f));
		reset();
	}

	string getName() const override { return "AccumulatePass"; }

	void render(CUDABuffer& frame);
	CUDABuffer& result() { return *mAccumBuffer; }

private:
	uint mAccumCount{ 0 };
	bool mMovingAverage{ false };
	uint mMaxAccumCount{ 0U };
	CUDABuffer *mAccumBuffer;
};

KRR_NAMESPACE_END