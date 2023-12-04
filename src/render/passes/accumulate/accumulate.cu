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

void AccumulatePass::render(RenderContext *context) {
	PROFILE("Accumulate pass");
	if (mScene->getChanges()) reset();
	static size_t lastResetFrame = 0;
	auto lastSceneUpdates = mScene->getSceneGraph()->getLastUpdateRecord();
	if (lastSceneUpdates.updateFlags != SceneGraphNode::UpdateFlags::None &&
		lastResetFrame < lastSceneUpdates.frameIndex) {
		reset();
		lastResetFrame = lastSceneUpdates.frameIndex;
	} 
	
	RGBA *accumBuffer = (RGBA *) mAccumBuffer->data();
	CudaRenderTarget currentBuffer = context->getColorTexture()->getCudaRenderTarget();
	GPUParallelFor(getFrameSize()[0] * getFrameSize()[1], KRR_DEVICE_LAMBDA(int i) {
		float currentWeight = 1.f / (mAccumCount + 1);
		RGBA currentPixel = currentBuffer.read(i);
		if (mAccumCount > 0) {
			if (mMode == Mode::MovingAverage) // moving average mode
				accumBuffer[i] =
					lerp(accumBuffer[i], currentPixel, currentWeight);
			else if (!mMaxAccumCount || mAccumCount < mMaxAccumCount) // sum mode
				accumBuffer[i] = accumBuffer[i] + currentPixel;
		} else {
			accumBuffer[i] = currentPixel;
		}
		if (mMode == Mode::MovingAverage)
			currentBuffer.write(accumBuffer[i], i);
		else
			currentBuffer.write(accumBuffer[i] * currentWeight, i);
	});
	if (!mMaxAccumCount || mAccumCount < mMaxAccumCount) {
		mAccumCount++;
		mCurrentTime = CpuTimer::getCurrentTimePoint();
	}
}

void AccumulatePass::resize(const Vector2i &size) {
	RenderPass::resize(size);
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