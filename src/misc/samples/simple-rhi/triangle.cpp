#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include "common/devicemanager.h"
#include "common/shader.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "Hello Triangle";

class BasicTriangle : public IRenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;

public:
	using IRenderPass::IRenderPass;

	bool Init() {
		ShaderLoader shaderLoader(GetDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/simple-rhi/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/simple-rhi/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);

		if (!m_VertexShader || !m_PixelShader) 
			return false;
		
		m_CommandList = GetDevice()->createCommandList();

		return true;
	}

	void BackBufferResizing() override { m_Pipeline = nullptr; }

	void Animate(float fElapsedTimeSeconds) override {
		GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
	}

	void Render(nvrhi::IFramebuffer *framebuffer) override {
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;

			m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
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
		GetDevice()->executeCommandList(m_CommandList);
	}
};


extern "C" int main(int argc, const char *argv[]) {
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);
	DeviceCreationParameters deviceParams;
	deviceParams.renderFormat				= nvrhi::Format::RGBA8_UNORM;
	deviceParams.swapChainBufferCount		= 2;
	deviceParams.maxFramesInFlight			= 1;
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}
	
	{
		BasicTriangle example(deviceManager);
		if (example.Init()) {
			deviceManager->AddRenderPassToBack(&example);
			deviceManager->RunMessageLoop();
			deviceManager->RemoveRenderPass(&example);
		}
	}

	deviceManager->Shutdown();
	delete deviceManager;
	return 0;
}

KRR_NAMESPACE_END