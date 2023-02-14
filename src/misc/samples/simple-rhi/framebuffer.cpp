#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include "common/devicemanager.h"
#include "common/textureloader.h"
#include "common/shader.h"
#include "common/cufriends.h"
#include "SineWaveSimulation.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "CuFramebuffer";

class BasicTriangle : public IRenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;
	cudaSurfaceObject_t m_cudaFrame{};
	std::shared_ptr<vkrhi::CudaVulkanFriend> m_CUFriend;
	int m_blocks, m_threads;
	float m_ElapsedTime{};

public:
	using IRenderPass::IRenderPass;

	bool Init() {
		ShaderLoader shaderLoader(GetDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/simple-rhi/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/simple-rhi/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);
		
		m_CUFriend = std::make_shared<vkrhi::CudaVulkanFriend>(GetDevice(false));

		if (!m_VertexShader || !m_PixelShader) 
			return false;
		
		m_CommandList = GetDevice()->createCommandList();

		return true;
	}

	void BackBufferResizing() override { 
		m_Pipeline = nullptr; 
		m_cudaFrame = {};
	}

	void Animate(float fElapsedTimeSeconds) override {
		GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
		m_ElapsedTime += fElapsedTimeSeconds;
	}

	void Render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer = frame->getFramebuffer();
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;

			m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

		auto fbInfo = framebuffer->getFramebufferInfo();
		
		/*m_CommandList->open();
		nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

		nvrhi::GraphicsState state;
		state.pipeline	  = m_Pipeline;
		state.framebuffer = framebuffer;
		state.viewport.addViewportAndScissorRect(fbInfo.getViewport());

		m_CommandList->setGraphicsState(state);

		nvrhi::DrawArguments args;
		args.vertexCount = 3;
		m_CommandList->draw(args);
		m_CommandList->close();
		GetDevice()->executeCommandList(m_CommandList);*/

		GetDevice()->waitForIdle();
		CUDA_SYNC_CHECK();
		//if (!m_cudaFrame) {
		//	m_cudaFrame = m_CUFriend->mapVulkanTextureToCudaSurface(
		//		framebuffer->getDesc().colorAttachments[0].texture, cudaArrayColorAttachment);
		//}
		cudaSurfaceObject_t cudaFrame = frame->getMappedCudaSurface(m_CUFriend);

		drawScreen(cudaFrame, m_ElapsedTime, fbInfo.width, fbInfo.height);
		CUDA_SYNC_CHECK();
	}
};


extern "C" int main(int argc, const char *argv[]) {
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);
	
	DeviceCreationParameters deviceParams		= {};
	deviceParams.enableDebugRuntime				= true;
	deviceParams.enableNvrhiValidationLayer		= true;
	deviceParams.swapChainFormat				= nvrhi::Format::SRGBA8_UNORM;
	deviceParams.renderFormat					= nvrhi::Format::RGBA8_UNORM;				
	deviceParams.enableCudaInterop				= true;

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