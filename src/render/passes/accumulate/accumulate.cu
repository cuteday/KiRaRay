#include "accumulate.h"

#include "util/math_utils.h"
#include "render/profiler/profiler.h"
#include "device/cuda.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

void AccumulatePass::reset() {
	mAccumCount = 0;
	mStartTime = mCurrentTime = CpuTimer::getCurrentTimePoint();
}

void AccumulatePass::render(CUDABuffer& frame) {
	if (!mEnable) return;
	PROFILE("Accumulate pass");
	if (mpScene->getChanges()) reset();
	Vector4f* accumBuffer = (Vector4f*)mAccumBuffer->data();
	Vector4f* currentBuffer = (Vector4f*)frame.data();
	GPUParallelFor(mFrameSize[0] * mFrameSize[1], KRR_DEVICE_LAMBDA(int i) {
		float currentWeight = 1.f / (mAccumCount + 1);
		if (mAccumCount > 0) {
			if (mMode == Mode::MovingAverage) // moving average mode
				accumBuffer[i] = utils::lerp(accumBuffer[i], currentBuffer[i], currentWeight);
			else if (!mMaxAccumCount || mAccumCount < mMaxAccumCount) // sum mode
				accumBuffer[i] = accumBuffer[i] + currentBuffer[i];
		} else {
			accumBuffer[i] = currentBuffer[i];
		}
		if (mMode == Mode::MovingAverage)
			currentBuffer[i] = accumBuffer[i];
		else
			currentBuffer[i] = accumBuffer[i] * currentWeight;
	});
	if (!mMaxAccumCount || mAccumCount < mMaxAccumCount) {
		mAccumCount++;
		mCurrentTime = CpuTimer::getCurrentTimePoint();
	}

}

void AccumulatePass::resize(const Vector2i &size) {
	mFrameSize = size;
	if (!mAccumBuffer)
		mAccumBuffer = new CUDABuffer();
	mAccumBuffer->resize(size[0] * size[1] * sizeof(Vector4f));
	reset();
}

void AccumulatePass::renderUI() {
	if (ui::Checkbox("Enabled", &mEnable))
		reset();
	if (mEnable) {
		static const char *modeNames[] = { "Accumulate", "Moving Average" };
		if (ui::Combo("Enable moving average", (int *) &mMode, modeNames, (int) Mode::Count))
			reset();
		ui::Text("Accumulate count: %d", mAccumCount);
		ui::Text("Elapsed time: %.2f", 
			CpuTimer::calcDuration(mStartTime, mCurrentTime) * 1e-3 );
		if (ui::DragInt("Max accum count", (int *) &mMaxAccumCount, 1, 0, 1e9))
			reset();
		if (ui::Button("reset")) reset();
	}
}

KRR_REGISTER_PASS_DEF(AccumulatePass);
KRR_NAMESPACE_END