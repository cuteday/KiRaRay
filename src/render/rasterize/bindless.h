#pragma once
#include "common.h"
#include "renderpass.h"

#include "vulkan/shader.h"
#include "vulkan/binding.h"
#include "vulkan/descriptor.h"
#include "vulkan/helperpass.h"

KRR_NAMESPACE_BEGIN

class RenderTargets;

class BindlessRender : public RenderPass {
public:
	struct ViewConstants {
		Matrix4f worldToView;
		Matrix4f viewToClip;
		Matrix4f worldToClip;
		Vector3f cameraPosition;
		int padding;
	};

	struct LightConstants {
		uint32_t numLights;
		Color3f ambientBottom;
		Color3f ambientTop;
		uint32_t padding;
	};

	enum class MSAA {
		NONE	 = 0, 
		MSAA_2X	 = 1,
		MSAA_4X	 = 2,
		MSAA_8X	 = 3,
	};
	
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<BindlessRender>;
	KRR_REGISTER_PASS_DEC(BindlessRender);

	void initialize() override;
	void render(RenderContext* context) override;
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
	vkrhi::BufferHandle mLightConstants;
	vkrhi::FramebufferHandle mFramebuffer;

	std::shared_ptr<ShaderLoader> mShaderLoader;
	std::shared_ptr<DescriptorTableManager> mDescriptorTableManager;
	std::shared_ptr<BindingCache> mBindingCache;
	std::shared_ptr<CommonRenderPasses> mHelperPass;
	std::unique_ptr<RenderTargets> mRenderTargets;

	MSAA mMSAA{MSAA::MSAA_8X};

	friend void to_json(json& j, const BindlessRender& p) {

	}

	friend void from_json(const json& j, BindlessRender& p) {

	}
};

KRR_NAMESPACE_END