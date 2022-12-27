#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"

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
	friend void to_json(json& j, const ToneMappingPass& p) {
		j = json{ { "exposure", p.mExposureCompensation }, { "operator", p.mOperator } };
	}
	
	friend void from_json(const json &j, ToneMappingPass &p) {
		j.at("exposure").get_to(p.mExposureCompensation);
		j.at("operator").get_to(p.mOperator);		
	}
	
	float mExposureCompensation{ 1 };
	Operator mOperator{ Operator::Linear };
};

KRR_ENUM_DEINFE(ToneMappingPass::Operator, { 
	{ ToneMappingPass::Operator::Linear, "linear" },
	{ ToneMappingPass::Operator::Reinhard, "reinhard" },
	{ ToneMappingPass::Operator::Aces, "aces" },
	{ ToneMappingPass::Operator::Uncharted2, "uncharted2" },
	{ ToneMappingPass::Operator::HejiHable, "hejihable" },
})

KRR_NAMESPACE_END