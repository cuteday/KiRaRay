#include "accumulate.h"

#include "math/utils.h"
#include "render/profiler/profiler.h"
#include "device/cuda.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

void AccumulatePass::render(CUDABuffer& frame) {
	if (!mEnable) return;
	PROFILE("Accumulate pass");
	if (mpScene->getChanges()) reset();
	Vec4f* accumBuffer = (Vec4f*)mAccumBuffer->data();
	Vec4f* currentBuffer = (Vec4f*)frame.data();
	mMovingAverage = mMaxAccumCount > 0;
	GPUParallelFor(mFrameSize[0] * mFrameSize[1], KRR_DEVICE_LAMBDA(int i) {
		float currentWeight = 1.f / (mAccumCount + 1);
		if (mAccumCount > 0) {
			if (mMovingAverage) // moving average mode
				accumBuffer[i] = utils::lerp(accumBuffer[i], currentBuffer[i], currentWeight);
			else				// sum mode
				accumBuffer[i] = accumBuffer[i] + currentBuffer[i];
		}
		else {
			accumBuffer[i] = currentBuffer[i];
		}
		if (mMovingAverage)
			currentBuffer[i] = accumBuffer[i];
		else
			currentBuffer[i] = accumBuffer[i] * currentWeight;
	});
	if (!mMaxAccumCount || mAccumCount < mMaxAccumCount)
		mAccumCount++;
}

void AccumulatePass::renderUI() {
	if (ui::CollapsingHeader("Accumulate pass")) {
		if (ui::Checkbox("Enabled", &mEnable))
			reset();
		if (mEnable) {
			ui::Text("Accumulate count: %d\n", mAccumCount);
			if (ui::DragInt("Max accum count", (int *) &mMaxAccumCount, 1, 0, 1e9))
				reset();
			if (ui::Button("reset"))
				reset();
		}
	}
}

KRR_NAMESPACE_END