#include "errormeasure.h"
#include "metrics.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
static const char *metricNames[] = { "MSE", "MAPE", "RelMSE" };
}

void ErrorMeasurePass::beginFrame(CUDABuffer &frame) {
	mNeedsEvaluate |= mContinuousEvaluate && (mFrameNumber % mEvaluateInterval == 0);
}

void ErrorMeasurePass::render(CUDABuffer &frame) {
	if (mNeedsEvaluate) {
		PROFILE("Metric calculation");
		CHECK_LOG(mReferenceImage.getSize() == mFrameSize,
				  "ErrorMeasure::Reference image size does not match frame size!");
		size_t n_elememts = mFrameSize[0] * mFrameSize[1];
		mResult =
			calculateMetric(mMetric, reinterpret_cast<Color4f *>(frame.data()),
							reinterpret_cast<Color4f *>(mReferenceImageBuffer.data()), n_elememts);
		if (mLogResults)
			Log(Info, "Evaluate result: %s = %f", metricNames[(int) mMetric], mResult);
		
		mNeedsEvaluate = false;
		mIsEvaluated   = true;
	}
}

void ErrorMeasurePass::endFrame(CUDABuffer &frame) {
	mFrameNumber++;
}

void ErrorMeasurePass::resize(const Vector2i &size) {
	RenderPass::resize(size);
}

void ErrorMeasurePass::renderUI() { 
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		if (ui::Combo("Metric", (int *) &mMetric, metricNames, (int)ErrorMetric::Count))
			mIsEvaluated = false;
		static char referencePath[256] = "";
		ui::InputText("Reference", referencePath, sizeof(referencePath));
		if (ui::Button("Load")) {
			loadReferenceImage(referencePath);
		}
		if (mReferenceImage.isValid()) {
			ui::Text("Reference image: %s", mReferenceImagePath.c_str());
			ui::Checkbox("Continuous evaluate", &mContinuousEvaluate);
			if (mContinuousEvaluate)
				ui::InputScalar("Evaluate every", ImGuiDataType_::ImGuiDataType_U64,
								&mEvaluateInterval);
			if (ui::Button("Evaluate")) mNeedsEvaluate = 1;
		}
		if (mIsEvaluated)
			ui::Text("%s: %f", metricNames[(int)mMetric], mResult);
	}
}

float ErrorMeasurePass::calculateMetric(ErrorMetric metric, 
	const Color4f* frame, const Color4f* reference, size_t n_elements) {
	return calc_metric(frame, reference, n_elements, metric);	
}

bool ErrorMeasurePass::loadReferenceImage(const string &path) {
 	bool success = mReferenceImage.loadImage(path, true, false);
	if (success) {
		// TODO: find out why saving an exr image yields this permutation on pixel format?
		mReferenceImage.permuteChannels(Vector4i{ 3, 0, 1, 2});
		mReferenceImageBuffer.resize(mReferenceImage.getSizeInBytes());
		mReferenceImageBuffer.copy_from_host(reinterpret_cast<Color4f*>(mReferenceImage.data()), 
			mReferenceImage.getSizeInBytes() / sizeof(Color4f));
		mIsEvaluated = mNeedsEvaluate = false;
		mReferenceImagePath			  = path;
	} else {
		Log(Error, "ErrorMeasure::Failed to load reference image from %s", path.c_str());
	}
	return success;
}

KRR_REGISTER_PASS_DEF(ErrorMeasurePass);
KRR_NAMESPACE_END