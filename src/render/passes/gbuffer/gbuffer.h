#pragma once
#include "common.h"
#include "renderpass.h"

KRR_NAMESPACE_BEGIN

class GBufferPass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<GBufferPass>;
	KRR_REGISTER_PASS_DEC(GBufferPass);
	GBufferPass() = default;
	
	void render(RenderContext *context);
	void renderUI() override;
	std::string getName() const override { return "GBufferPass"; }

private:
	bool mEnableDepth{};
	bool mEnableDiffuse{};
	bool mEnableSpecular{};
	bool mEnableNormal{};
	bool mEnableEmissive{};
	bool mEnableMotion{};
};

KRR_NAMESPACE_END