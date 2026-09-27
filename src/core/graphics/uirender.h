#pragma once

#include <array>

#include <common.h>
#include <input.h>
#include <renderpass.h>
#include <graphics/shader.h>

#include <imgui.h>
#include <nvrhi/nvrhi.h>

NAMESPACE_BEGIN(krr)

class UIRenderer: public RenderPass {
private:
	ImGuiContext *mContext = nullptr;
	ImGuiStyle mBaseStyle;
	float mFontScale = 0.f;
	float mFramebufferScale = 0.f;
	float mPreviousTime = -1.f;

	nvrhi::DeviceHandle device;
	nvrhi::CommandListHandle m_commandList;

	nvrhi::ShaderHandle vertexShader;
	nvrhi::ShaderHandle pixelShader;
	nvrhi::InputLayoutHandle shaderAttribLayout;

	nvrhi::TextureHandle fontTexture;
	nvrhi::SamplerHandle fontSampler;

	nvrhi::BufferHandle vertexBuffer;
	nvrhi::BufferHandle indexBuffer;

	nvrhi::BindingLayoutHandle bindingLayout;
	nvrhi::GraphicsPipelineDesc basePSODesc;

	nvrhi::GraphicsPipelineHandle pso;
	std::unordered_map<nvrhi::ITexture *, nvrhi::BindingSetHandle>
		bindingsCache;

	std::vector<ImDrawVert> vtxBuffer;
	std::vector<ImDrawIdx> idxBuffer;

	std::array<bool, GLFW_KEY_LAST + 1> keyDown = {false};

public:
	using RenderPass::RenderPass;
	UIRenderer() = default;
	UIRenderer(const UIRenderer &) = delete;
	UIRenderer &operator=(const UIRenderer &) = delete;
	bool isCudaPass() const override { return false; }
	using SharedPtr = std::shared_ptr<UIRenderer>;
	~UIRenderer();
	string getName() const override { return "UIRenderer"; }

	void initialize() override;
	void tick(float elapsedTimeSeconds) override;
	void beginFrame(RenderContext* context) override;
	void endFrame(RenderContext* context) override;
	void render(RenderContext *context) override;
	void resizing() override;

	virtual bool onMouseEvent(const io::MouseEvent &mouseEvent) override;
	virtual bool onKeyEvent(const io::KeyboardEvent &keyEvent) override;
	void onWindowFocus(int focused) override;

protected:
	bool reallocateBuffer(nvrhi::BufferHandle &buffer, size_t requiredSize,
						  size_t reallocateSize, bool isIndexBuffer);

	bool createFontTexture(nvrhi::ICommandList *commandList);
	void updateFont(nvrhi::ICommandList *commandList, float fontScale, float framebufferScale);

	nvrhi::IGraphicsPipeline *getPSO(nvrhi::IFramebuffer *fb);
	nvrhi::IBindingSet *getBindingSet(nvrhi::ITexture *texture);
	bool updateGeometry(nvrhi::ICommandList *commandList);
};

NAMESPACE_END(krr)
