#include "accumulate.h"

#include "math/utils.h"
#include "device/cuda.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

void AccumulatePass::render(CUDABuffer& frame) {
	if (!mEnable) return;
	if (mpScene->getChanges()) reset();
	vec4f* accumBuffer = (vec4f*)mAccumBuffer->data();
	vec4f* currentBuffer = (vec4f*)frame.data();
	GPUParallelFor(mFrameSize.x * mFrameSize.y, KRR_DEVICE_LAMBDA(int i) {
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
	mAccumCount = min(mAccumCount + 1, mMaxAccumCount - 1);
}

KRR_NAMESPACE_END