#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "texture.h"
#include "device/timer.h"
#include "device/context.h"
#include "device/buffer.h"

#include "renderpass.h"
#include "metrics.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class ErrorMeasurePass : public RenderPass {
public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<ErrorMeasurePass>;
	KRR_REGISTER_PASS_DEC(ErrorMeasurePass);

	ErrorMeasurePass() = default;
	~ErrorMeasurePass() = default;
	void beginFrame(RenderContext* context) override;
	void render(RenderContext* context) override;
	void renderUI() override;
	void resize(const Vector2i& size) override;
	void finalize() override;

	string getName() const override { return "ErrorMeasurePass"; }

protected:
	typedef struct {
		size_t timestep;
		double timepoint;
		json metrics;
	} EvaluationData;

	void reset();
	bool loadReferenceImage(const string &path);
	
	std::shared_ptr<Image> mReferenceImage;
	TypedBuffer<RGBA> mReferenceImageBuffer;
	bool mShowReferenceImage{false};
	ErrorMetric mMetric{ ErrorMetric::RelMSE };
	json mLastResult;
	string mReferenceImagePath;
	bool mNeedsEvaluate{}, mContinuousEvaluate{};
	bool mLogResults{}, mSaveResults{};
	size_t mFrameNumber{ 0 }, mEvaluateInterval{ 1 };
	std::vector<EvaluationData> mEvaluationResults;
	CpuTimer::TimePoint mStartTime;

	bool mShowPixelError{false};
	bool mJetColorMapOn{false};
	float mJetColorMapVMax{0.01f};
	float mJetColorMapVMaxAdjusted{0.05f};

	friend void to_json(json &j, const ErrorMeasurePass &p) {
		j = json{{"metric", p.mMetric},
				 {"reference", p.mReferenceImagePath},
				 {"continuous", p.mContinuousEvaluate},
				 {"interval", p.mEvaluateInterval},
				 {"log", p.mLogResults},
				 {"save", p.mSaveResults},
				 {"vJetMax", p.mJetColorMapVMax},
				 {"vJetMaxAdjusted", p.mJetColorMapVMax}};
	}

	friend void from_json(const json &j, ErrorMeasurePass &p) {
		p.mMetric				   = j.value("metric", ErrorMetric::RelMSE);
		p.mContinuousEvaluate	   = j.value("continuous", false);
		p.mEvaluateInterval		   = j.value("interval", 1);
		p.mLogResults			   = j.value("log", false);
		p.mSaveResults			   = j.value("save", false);
		p.mJetColorMapVMax		   = j.value("vJetMax", 0.01f);
		p.mJetColorMapVMaxAdjusted = j.value("vJetMaxAdjusted", 0.05f);

		if (gpContext->getGlobalConfig().contains("reference"))
			p.loadReferenceImage(gpContext->getGlobalConfig().at("reference"));
		if (j.contains("reference")) p.loadReferenceImage(j.at("reference"));
	}
};

NAMESPACE_END(krr)