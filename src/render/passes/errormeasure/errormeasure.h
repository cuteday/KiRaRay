#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "texture.h"
#include "device/context.h"
#include "device/buffer.h"

#include "renderpass.h"
#include "metrics.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

class ErrorMeasurePass : public RenderPass {
public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<ErrorMeasurePass>;
	KRR_REGISTER_PASS_DEC(ErrorMeasurePass);

	ErrorMeasurePass() = default;
	~ErrorMeasurePass() = default;
	void beginFrame(CUDABuffer &frame) override;
	void render(CUDABuffer &frame) override;
	void endFrame(CUDABuffer &frame) override;
	void renderUI() override;
	void resize(const Vector2i& size) override;

	string getName() const override { return "ErrorMeasurePass"; }

protected:
	bool loadReferenceImage(const string &path);
	static float calculateMetric(ErrorMetric metric, 
		const Color4f *frame, const Color4f *reference, size_t n_elements);
	Image mReferenceImage;
	TypedBuffer<Color4f> mReferenceImageBuffer;
	ErrorMetric mMetric{ ErrorMetric::RelMSE };
	float mResult;
	string mReferenceImagePath;
	bool mIsEvaluated{}, mNeedsEvaluate{}, mContinuousEvaluate{};
	bool mLogResults{};
	size_t mFrameNumber{ 0 }, mEvaluateInterval{ 1 };

	friend void to_json(json &j, const ErrorMeasurePass &p) { 
		j = json{ 
			{ "metric", p.mMetric }, 
			{ "reference", p.mReferenceImagePath },
			{ "continuous", p.mContinuousEvaluate },
			{ "interval", p.mEvaluateInterval }, 
			{ "log", p.mLogResults }
		};
	}

	friend void from_json(const json &j, ErrorMeasurePass &p) {
		p.mMetric			  = j.value("metric", ErrorMetric::RelMSE);
		p.mContinuousEvaluate = j.value("continuous", false);
		p.mEvaluateInterval	  = j.value("interval", 1);
		p.mLogResults		  = j.value("log", false);
		if (gpContext->getGlobalConfig().contains("reference"))
			p.loadReferenceImage(gpContext->getGlobalConfig().at("reference"));
		if (j.contains("reference"))
			p.loadReferenceImage(j.at("reference"));
	}
};

KRR_NAMESPACE_END