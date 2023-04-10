#pragma once
#include "common.h"
#include "renderpass.h"

#include "vulkan/shader.h"
#include "vulkan/binding.h"
#include "vulkan/descriptor.h"
#include "vulkan/helperpass.h"

KRR_NAMESPACE_BEGIN

class BindlessRender : public RenderPass {
public:
	struct ViewConstants {
		Matrix4f worldToView;
		Matrix4f viewToClip;
		Matrix4f worldToClip;
	};
	
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<BindlessRender>;
	KRR_REGISTER_PASS_DEC(BindlessRender);

	void initialize() override;
	void render(RenderFrame::SharedPtr frame) override;
	void renderUI() override;
	void resize(const Vector2i &size) override;

	string getName() const override { return "BindlessRender"; }

private:
	vkrhi::CommandListHandle mCommandList;
	vkrhi::BindingLayoutHandle mBindingLayout;
	vkrhi::BindingLayoutHandle mBindlessLayout;
	vkrhi::BindingSetHandle mBindingSet;
	vkrhi::ShaderHandle mVertexShader;
	vkrhi::ShaderHandle mPixelShader;
	vkrhi::GraphicsPipelineHandle mGraphicsPipeline;

	vkrhi::BufferHandle mViewConstants;
	vkrhi::TextureHandle mDepthBuffer;
	vkrhi::TextureHandle mColorBuffer;
	vkrhi::FramebufferHandle mFramebuffer;

	std::shared_ptr<ShaderLoader> mShaderLoader;
	std::shared_ptr<DescriptorTableManager> mDescriptorTableManager;
	std::shared_ptr<BindingCache> mBindingCache;
	std::shared_ptr<CommonRenderPasses> mHelperPass;

	friend void to_json(json& j, const BindlessRender& p) {

	}

	friend void from_json(const json& j, BindlessRender& p) {

	}
};

KRR_NAMESPACE_END