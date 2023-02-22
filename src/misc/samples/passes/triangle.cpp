#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include <main/renderer.h>
#include <vulkan/shader.h>
#include <renderpass.h>

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "Hello Triangle";

class BasicTriangle : public RenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;

public:
	using RenderPass::RenderPass;

	void initialize() {
		ShaderLoader shaderLoader(getVulkanDevice());
		m_VertexShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);

		if (!m_VertexShader || !m_PixelShader)
			Log(Fatal, "Shader initialization failed");
		m_CommandList = getVulkanDevice()->createCommandList();
	}

	void resizing() override { m_Pipeline = nullptr; }

	void tick(float fElapsedTimeSeconds) override {}

	void render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer = frame->getFramebuffer();
		
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;

			m_Pipeline = getVulkanDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

		m_CommandList->open();

		nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

		nvrhi::GraphicsState state;
		state.pipeline	  = m_Pipeline;
		state.framebuffer = framebuffer;
		state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());

		m_CommandList->setGraphicsState(state);

		nvrhi::DrawArguments args;
		args.vertexCount = 3;
		m_CommandList->draw(args);
		m_CommandList->close();
		getVulkanDevice()->executeCommandList(m_CommandList);
	}
};

extern "C" int main(int argc, const char *argv[]) {
	auto app = std::make_unique<DeviceManager>();
	DeviceCreationParameters deviceParams;
	deviceParams.renderFormat				= nvrhi::Format::RGBA8_UNORM;
	deviceParams.swapChainBufferCount		= 2;
	deviceParams.maxFramesInFlight			= 1;
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!app->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}
	{
		auto example = std::make_shared<BasicTriangle>(app.get());
		example->initialize();
		app->AddRenderPassToBack(example);
		app->RunMessageLoop();
		app->RemoveRenderPass(example);
	}
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END