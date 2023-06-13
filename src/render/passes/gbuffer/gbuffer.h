#pragma once
#include "common.h"
#include "device.h"
#include "renderpass.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

class GBufferPass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<GBufferPass>;
	KRR_REGISTER_PASS_DEC(GBufferPass);
	GBufferPass() = default;
	
	void initialize() override;
	void setScene(Scene::SharedPtr scene);
	void render(RenderContext *context);
	void renderUI() override;
	std::string getName() const override { return "GBufferPass"; }

private:
	OptixBackend::SharedPtr mOptixBackend;
	LaunchParamsGBuffer mLaunchParams;

	bool mEnableDepth{};
	bool mEnableDiffuse{};
	bool mEnableSpecular{};
	bool mEnableNormal{};
	bool mEnableEmissive{};
	bool mEnableMotion{};
};

KRR_NAMESPACE_END