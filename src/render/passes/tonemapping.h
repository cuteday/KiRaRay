#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"
#include "math/math.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class ToneMappingPass: public RenderPass {
public:
	using SharedPtr = std::shared_ptr<ToneMappingPass>;

	enum class Operator {
		Linear = 0,
		Reinhard,
		Aces,
		Uncharted2,
		HejiHable,
		NumsOperators,
	};

	ToneMappingPass() = default;

	void renderUI() override {
		static const char* operators[] = { "Linear", "Reinhard", "Aces", "Uncharted2", "HejiHable" };
		if (ui::CollapsingHeader("Tone mapping pass")) {
			ui::Checkbox("Enabled", &mEnable);
			if (mEnable) {
				ui::SliderFloat("Exposure compensation", &mExposureCompensation, 0.001, 100, "%.3f");
				ui::Combo("Tonemap operator", (int*)&mOperator, operators, (int)Operator::NumsOperators);
			}
		}
	}

	void setOperator(Operator toneMappingOperator)
		{mOperator = toneMappingOperator; }
	Operator getOperator() const { return mOperator; }
	void render(CUDABuffer& frame);

private:
	bool mEnable{ true };
	float mExposureCompensation{ 1 };
	Operator mOperator{ Operator::Linear };
};

KRR_NAMESPACE_END