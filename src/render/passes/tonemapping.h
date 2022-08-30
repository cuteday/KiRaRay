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
	KRR_REGISTER_PASS_DEC(ToneMappingPass);

	enum class Operator {
		Linear = 0,
		Reinhard,
		Aces,
		Uncharted2,
		HejiHable,
		NumsOperators,
	};

	ToneMappingPass() = default;

	void renderUI() override;

	void setOperator(Operator toneMappingOperator)
		{mOperator = toneMappingOperator; }
	Operator getOperator() const { return mOperator; }
	void render(CUDABuffer& frame) override;

	string getName() const override { return "ToneMappingPass"; }

private:
	bool mEnable{ true };
	float mExposureCompensation{ 1 };
	Operator mOperator{ Operator::Linear };
};

KRR_NAMESPACE_END