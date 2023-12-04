#include "errormeasure.h"
#include "metrics.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
static const char *metricNames[] = { "MSE", "MAPE", "SMAPE", "RelMSE" };
}

void ErrorMeasurePass::beginFrame(RenderContext* context) {
	if (!mFrameNumber) reset();
	mFrameNumber++;
	mNeedsEvaluate |= mContinuousEvaluate && (mFrameNumber % mEvaluateInterval == 0);
}

void ErrorMeasurePass::render(RenderContext *context) {
	PROFILE("Metric calculation");
	if (mNeedsEvaluate && mReferenceImage && mReferenceImage->isValid()) {
		CHECK_LOG(mReferenceImage->getSize() == getFrameSize(),
				  "ErrorMeasure::Reference image size does not match frame size!");
		size_t n_elememts = getFrameSize()[0] * getFrameSize()[1];
		float result	  = calc_metric(context->getColorTexture()->getCudaRenderTarget(),
			reinterpret_cast<RGBA *>(mReferenceImageBuffer.data()),
			n_elememts, mMetric);
		mLastResult = { { string(metricNames[(int) mMetric]), result } };
		if (mLogResults)
			Log(Info, "Evaluating frame #%zd: %s", mFrameNumber, mLastResult.dump().c_str());
		if (mSaveResults)
			mEvaluationResults.push_back(
					{ mFrameNumber,
					CpuTimer::calcElapsedTime(mStartTime) * 1e-3,
					mLastResult });
		mNeedsEvaluate = false;
	}
}

void ErrorMeasurePass::resize(const Vector2i &size) {
	RenderPass::resize(size); }

void ErrorMeasurePass::finalize() {
	if (mSaveResults) {
		string output_name = gpContext->getGlobalConfig().contains("name")
						 ? gpContext->getGlobalConfig()["name"] : "result";
		fs::path save_path = File::outputDir() / "error" / (output_name + ".json");
		json timesteps, timepoints, data, result;
		for (const EvaluationData &e : mEvaluationResults) {
			timesteps.push_back((int)e.timestep);
			timepoints.push_back((float) e.timepoint);
			data.push_back(e.metrics);
		}
		result["timesteps"] = timesteps, 
		result["timepoints"] = timepoints,
		result["data"] = data;
		File::saveJSON(save_path, result);
		logInfo("Saved error evaluation data to " + save_path.string());
	}
}

void ErrorMeasurePass::renderUI() { 
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		if (ui::Combo("Metric", (int *) &mMetric, metricNames, (int)ErrorMetric::Count))
			reset();
		static char referencePath[256] = "";
		ui::InputText("Reference", referencePath, sizeof(referencePath));
		if (ui::Button("Load")) {
			loadReferenceImage(referencePath);
		}
		if (mReferenceImage && mReferenceImage->isValid()) {
			ui::Text("Reference image: %s", mReferenceImagePath.c_str());
			ui::Checkbox("Continuous evaluate", &mContinuousEvaluate);
			if (mContinuousEvaluate)
				ui::InputScalar("Evaluate every", ImGuiDataType_::ImGuiDataType_U64,
								&mEvaluateInterval);
			if (ui::Button("Evaluate")) mNeedsEvaluate = 1;
		}
		if (!mLastResult.empty())
			ui::Text("%s", mLastResult.dump().c_str());
		ui::Checkbox("Log results", &mLogResults);
		ui::Checkbox("Save results", &mSaveResults);
	}
}

void ErrorMeasurePass::reset() {
	mFrameNumber   = 0;
	mNeedsEvaluate = false;
	mLastResult	   = {};
	mStartTime	   = CpuTimer::getCurrentTimePoint();
}

bool ErrorMeasurePass::loadReferenceImage(const string &path) {
	mReferenceImage = std::make_shared<Image>();
 	bool success = mReferenceImage->loadImage(path, true, false);
	if (success) {
		// TODO: find out why saving an exr image yields this permutation on pixel format?
		// This should be deleted once new reference images are updated.
		auto permute = [](auto pixel) {
			Array4i p = { 3, 0, 1, 2 };
			auto res  = pixel;
			for (int c = 0; c < 4; c++) 
				res[c] = pixel[p[c]];
			return res;
		};
		mReferenceImage->process(permute);
		mReferenceImageBuffer.resize(mReferenceImage->getSizeInBytes());
		mReferenceImageBuffer.copy_from_host(reinterpret_cast<RGBA*>(mReferenceImage->data()), 
			mReferenceImage->getSizeInBytes() / sizeof(RGBA));
		reset();
		mReferenceImagePath = path;
		Log(Info, "ErrorMeasure::Loaded reference image from %s.", path.c_str());
	} else {
		Log(Error, "ErrorMeasure::Failed to load reference image from %s", path.c_str());
	}
	return success;
}

KRR_REGISTER_PASS_DEF(ErrorMeasurePass);
KRR_NAMESPACE_END