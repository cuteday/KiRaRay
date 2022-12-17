#include "errormeasure.h"
#include "metrics.h"

KRR_NAMESPACE_BEGIN

void ErrorMeasurePass::render(CUDABuffer &frame) {
	if (mNeedsEvaluate) {
		mResult = calculateMetric(mMetric, reinterpret_cast<Color4f*>(frame.data()), 
			reinterpret_cast<Color4f *>(mReferenceImageBuffer.data()));
	}
}

void ErrorMeasurePass::resize(const Vector2i &size) {
	mFrameSize = size;
	mReferenceImageBuffer.resize(sizeof(Color4f) * size[0] * size[1]);
}

void ErrorMeasurePass::renderUI() { 
	static char *metricNames[] = { "MSE", "MAPE", "RelMSE" }; 
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		ui::Combo("Metric", (int *) &mMetric, metricNames, Metric::NumMetrics);
		static char referencePath[500] = "";
		ui::InputText("Reference path: ", referencePath, sizeof(referencePath));
		if (ui::Button("Load")) {
			loadReferenceImage(referencePath);
		}
		if (mReferenceImage.isValid()) {
			ui::Text("Reference image: %s", mReferenceImagePath.c_str());
			if (ui::Button("Evaluate"))
				mNeedsEvaluate = 1;
		}
		if (mIsEvaluated)
			ui::Text("%s: %f", metricNames[mMetric], mResult);
		
	}
}

float ErrorMeasurePass::calculateMetric(Metric metric, 
	const Color4f* frame, const Color4f* reference) {
	return 0;
}

bool ErrorMeasurePass::loadReferenceImage(const string &path) {
 	bool success = mReferenceImage.loadImage(path, false);
	if (success) {
		mReferenceImageBuffer.resize(mReferenceImage.getSizeInBytes());
		mReferenceImageBuffer.copy_from_host(mReferenceImage.data(),
											 mReferenceImage.getSizeInBytes());
		mIsEvaluated = mNeedsEvaluate = false;
		mReferenceImagePath			  = path;
	} else {
		Log(Error, "ErrorMeasure::Failed to load reference image from %s", path.c_str());
	}
	return success;
}

KRR_REGISTER_PASS_DEF(ErrorMeasurePass);
KRR_NAMESPACE_END