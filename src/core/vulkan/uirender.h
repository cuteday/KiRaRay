#pragma once

#include <array>

#include <common.h>
#include <input.h>
#include <renderpass.h>
#include <vulkan/shader.h>

#include <imgui.h>
#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

class UIRenderer: public RenderPass {
private:
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

	std::array<bool, 3> mouseDown				= {false};
	std::array<bool, GLFW_KEY_LAST + 1> keyDown = {false};

public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<UIRenderer>;
	~UIRenderer() { ImGui::DestroyContext(); }
	string getName() const override { return "UIRenderer"; }

	void initialize() override;
	void tick(float elapsedTimeSeconds) override;
	void beginFrame(RenderContext* context) override;
	void endFrame(RenderContext* context) override;
	void render(RenderContext *context) override;
	void resizing() override;

	virtual bool onMouseEvent(const io::MouseEvent &mouseEvent) override;
	virtual bool onKeyEvent(const io::KeyboardEvent &keyEvent) override;

protected:
	bool reallocateBuffer(nvrhi::BufferHandle &buffer, size_t requiredSize,
						  size_t reallocateSize, bool isIndexBuffer);

	bool createFontTexture(nvrhi::ICommandList *commandList);

	nvrhi::IGraphicsPipeline *getPSO(nvrhi::IFramebuffer *fb);
	nvrhi::IBindingSet *getBindingSet(nvrhi::ITexture *texture);
	bool updateGeometry(nvrhi::ICommandList *commandList);
};

KRR_NAMESPACE_END