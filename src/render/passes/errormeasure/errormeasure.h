#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "texture.h"
#include "device/buffer.h"

#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

class ErrorMeasurePass : public RenderPass {
public:
	enum Metric {
		MSE,
		RMSE,
		MAPE,
		RelMSE,
		NumMetrics
	};

	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<ErrorMeasurePass>;
	KRR_REGISTER_PASS_DEC(ErrorMeasurePass);
	KRR_CLASS_DEFINE(ErrorMeasurePass, mMetric);

	void render(CUDABuffer &frame) override;
	void renderUI() override;
	void resize(const Vector2i& size) override;

	string getName() const override { return "ErrorMeasurePass"; }

protected:
	bool loadReferenceImage(const string &path);
	static float calculateMetric(Metric metric, 
		const Color4f *frame, const Color4f *reference, size_t n_elements);
	Image mReferenceImage;
	TypedBuffer<Color4f> mReferenceImageBuffer;
	Metric mMetric{ MSE };
	float mResult;
	string mReferenceImagePath;
	bool mIsEvaluated{}, mNeedsEvaluate{};
};

KRR_INSTANTIATE_PASS(ErrorMeasurePass);
KRR_NAMESPACE_END