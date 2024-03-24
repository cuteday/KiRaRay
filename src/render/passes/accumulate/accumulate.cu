#include <filesystem>
#include "file.h"
#include "accumulate.h"

#include "util/math_utils.h"
#include "render/profiler/profiler.h"
#include "device/cuda.h"
#include "device/context.h"

NAMESPACE_BEGIN(krr)

size_t AccumulatePass::getPixelSize() const {
	switch (mPrecision) {
		case AccumulatePass::Precision::Float: return sizeof(Vector4f);
		case AccumulatePass::Precision::Double: return sizeof(Vector4d);
		default: return sizeof(Vector4f);
	}
}

void AccumulatePass::reset() {
	mAccumCount = 0;
	if (!mAccumBuffer) mAccumBuffer = new CUDABuffer();
	mAccumBuffer->resize(getFrameSize()[0] * getFrameSize()[1] * getPixelSize());
	mTask.reset();
}

template <typename DType>
void acculumate(Array4<DType> *accumBuffer, CudaRenderTarget currentBuffer, size_t nPixels, 
	size_t accumCount, size_t maxAccumCount, AccumulatePass::Mode mode) {
	GPUParallelFor(nPixels, [=] KRR_DEVICE (int i) mutable {
			DType currentWeight = static_cast<DType>(1) / (accumCount + 1);
			Array4<DType> currentPixel	= currentBuffer.read(i).cast<DType>();
			if (accumCount > 0) {
				if (mode == AccumulatePass::Mode::MovingAverage) // moving average mode
					accumBuffer[i] = lerp(accumBuffer[i], currentPixel, currentWeight);
				else if (!maxAccumCount || accumCount < maxAccumCount) // sum mode
					accumBuffer[i] = accumBuffer[i] + currentPixel;
			} else {
				accumBuffer[i] = currentPixel;
			}
			if (mode == AccumulatePass::Mode::MovingAverage)
				currentBuffer.write(accumBuffer[i].template cast<float>(), i);
			else
				currentBuffer.write((accumBuffer[i] * currentWeight).template cast<float>(), i);
		}, KRR_DEFAULT_STREAM);
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
	
	if (mPrecision == Precision::Float) {
		acculumate(reinterpret_cast<Array4<float> *>(accumBuffer), currentBuffer, 
			getFrameSize()[0] * getFrameSize()[1], mAccumCount, mMaxAccumCount, mMode);
	} else if (mPrecision == Precision::Double) {
		acculumate(reinterpret_cast<Array4<double> *>(accumBuffer), currentBuffer, 
			getFrameSize()[0] * getFrameSize()[1], mAccumCount, mMaxAccumCount, mMode);
	} 
	if (!mMaxAccumCount || mAccumCount < mMaxAccumCount) {
		mTask.tickFrame();
		mAccumCount++;
	}
}

void AccumulatePass::endFrame(RenderContext *context) {
	if (mTask.getBudgetType() != BudgetType::None && mTask.isFinished() && mExitOnFinish) 
		gpContext->requestExit();
}

void AccumulatePass::resize(const Vector2i &size) {
	RenderPass::resize(size);
	reset();
}

void AccumulatePass::finalize() {
	if (mSaveOnFinish) {
		cudaDeviceSynchronize();
		string outputName = gpContext->getGlobalConfig().contains("name")
								 ? gpContext->getGlobalConfig()["name"]
								 : "result";
		fs::path savePath = File::outputDir() / (outputName + ".exr");
		Image frame(getFrameSize(), Image::Format::RGBAfloat, false);
		size_t nPixels = getFrameSize()[0] * getFrameSize()[1]; 
		if (mPrecision == Precision::Float)
			mAccumBuffer->copy_to_host(frame.data(), nPixels * sizeof(RGBA));
		else if (mPrecision == Precision::Double) {
			CUDABuffer tmpBuffer(nPixels * getPixelSize());
			thrust::transform(thrust::device, reinterpret_cast<Array4d *>(mAccumBuffer->data()),
							  reinterpret_cast<Array4d *>(mAccumBuffer->data()) + nPixels,
							  reinterpret_cast<RGBA *>(tmpBuffer.data()),
							  [] KRR_DEVICE(const Array4d &d) -> RGBA { return d.cast<float>(); });
			tmpBuffer.copy_to_host(frame.data(), nPixels * sizeof(RGBA));
		}
		frame.saveImage(savePath, true);
	}
}

void AccumulatePass::renderUI() {
	if (ui::Checkbox("Enabled", &mEnable))
		reset();
	if (mEnable) {
		static const char *modeNames[] = { "Accumulate", "Moving Average" };
		static const char *precisionNames[] = {"Float", "Double"};
		if (ui::Combo("Accumulate mode", (int *) &mMode, modeNames, (int) Mode::Count))
			reset();
		if (ui::Combo("Precision", (int *) &mPrecision, precisionNames, (int) Precision::Count))
			reset();
		ui::Text("Accumulate count: %d", mAccumCount);
		ui::Text("Elapsed time: %.2f", mTask.getElapsedTime());
		if (ui::DragInt("Max accum count", (int *) &mMaxAccumCount, 1, 0, 1e9))
			reset();
		if (ui::Button("reset")) reset();
		if (mTask.getBudgetType() != BudgetType::None && ui::CollapsingHeader("Task progress")) 
			mTask.renderUI();
	}
}

KRR_REGISTER_PASS_DEF(AccumulatePass);
NAMESPACE_END(krr)