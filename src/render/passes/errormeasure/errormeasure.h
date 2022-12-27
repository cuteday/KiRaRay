#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "texture.h"
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
	void render(CUDABuffer &frame) override;
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
	bool mIsEvaluated{}, mNeedsEvaluate{};

	friend void to_json(json &j, const ErrorMeasurePass &p) { 
		j = json{ 
			{ "metric", p.mMetric }, 
			{ "reference", p.mReferenceImagePath }
		};
	}

	friend void from_json(const json &j, ErrorMeasurePass &p) {
		p.mMetric = j.value("metric", ErrorMetric::RelMSE);
		if (j.contains("reference"))
			p.loadReferenceImage(j.at("reference"));
	}
};

KRR_NAMESPACE_END