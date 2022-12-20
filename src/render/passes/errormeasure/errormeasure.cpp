#include "errormeasure.h"
#include "metrics.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

void ErrorMeasurePass::render(CUDABuffer &frame) {
	if (mNeedsEvaluate) {
		PROFILE("Metric calculation");
		size_t n_elememts = mFrameSize[0] * mFrameSize[1];
		mResult =
			calculateMetric(mMetric, reinterpret_cast<Color4f *>(frame.data()),
							reinterpret_cast<Color4f *>(mReferenceImageBuffer.data()), n_elememts);

		mNeedsEvaluate = false;
		mIsEvaluated   = true;
	}
}

void ErrorMeasurePass::resize(const Vector2i &size) {
	mFrameSize = size;
	mReferenceImageBuffer.resize(sizeof(Color4f) * size[0] * size[1]);
}

void ErrorMeasurePass::renderUI() { 
	static char *metricNames[]	   = { "MSE", "RMSE", "MAPE", "RelMSE" }; 
	static bool continuousEvaluate = false;
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		if(ui::Combo("Metric", (int *) &mMetric, metricNames, Metric::NumMetrics))
			mIsEvaluated = false;
		static char referencePath[256] = "";
		ui::InputText("Reference", referencePath, sizeof(referencePath));
		if (ui::Button("Load")) {
			loadReferenceImage(referencePath);
		}
		if (mReferenceImage.isValid()) {
			ui::Text("Reference image: %s", mReferenceImagePath.c_str());
			ui::Checkbox("Continuous evaluate", &continuousEvaluate);
			if (!continuousEvaluate && ui::Button("Evaluate")) 
				mNeedsEvaluate = 1;
			mNeedsEvaluate |= continuousEvaluate;
		}
		if (mIsEvaluated)
			ui::Text("%s: %f", metricNames[mMetric], mResult);
	}
}

float ErrorMeasurePass::calculateMetric(Metric metric, 
	const Color4f* frame, const Color4f* reference, size_t n_elements) {
	switch (metric) { 
	case Metric::MSE:
		return calc_metric_mse(frame, reference, n_elements);	
	case Metric::RMSE:
		return calc_metric_rmse(frame, reference, n_elements);	
	case Metric::MAPE:
		return calc_metric_mape(frame, reference, n_elements);	
	case Metric::RelMSE:
		return calc_metric_relmse(frame, reference, n_elements);	
	default:
		Log(Error, "ErrorMeasure::Unimplemented error metric!");
		return NAN;
	}
}

bool ErrorMeasurePass::loadReferenceImage(const string &path) {
 	bool success = mReferenceImage.loadImage(path, true, false);
	if (success) {
		CHECK_LOG(mReferenceImage.getSize() == mFrameSize, 
			"ErrorMeasure::Reference image size does not match frame size!");
		// TODO: find out why saving an exr image yields this permutation on pixel format?
		mReferenceImage.permuteChannels(Vector4i{ 3, 0, 1, 2});
		mReferenceImageBuffer.resize(mReferenceImage.getSizeInBytes());
		mReferenceImageBuffer.copy_from_host(reinterpret_cast<Color4f*>(mReferenceImage.data()), 
			mFrameSize[0] * mFrameSize[1]);
		mIsEvaluated = mNeedsEvaluate = false;
		mReferenceImagePath			  = path;
	} else {
		Log(Error, "ErrorMeasure::Failed to load reference image from %s", path.c_str());
	}
	return success;
}

KRR_REGISTER_PASS_DEF(ErrorMeasurePass);
KRR_NAMESPACE_END