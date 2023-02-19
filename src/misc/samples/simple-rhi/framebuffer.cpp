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
	std::shared_ptr<vkrhi::CudaVulkanFriend> m_CUFriend;
	std::unique_ptr<CommonRenderPasses> m_HelperPass;
	nvrhi::TextureHandle m_DrawTexture;
	cudaSurfaceObject_t m_DrawCudaSurface;
	std::unique_ptr<BindingCache> m_BindingCache;
	float m_ElapsedTime{};
	CUstream m_CudaStream{};
	vk::Semaphore m_cudaUpdateVkSem, m_VkUpdateCudaSem;

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
		m_HelperPass   = std::make_unique<CommonRenderPasses>(GetDevice());
		m_BindingCache = std::make_unique<BindingCache>(GetDevice());

		return true;
	}

	void BackBufferResizing() override { 
		m_Pipeline = nullptr; 
		m_DrawTexture = nullptr;
		m_DrawCudaSurface = 0;
	}

	void Animate(float fElapsedTimeSeconds) override {
		GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
		m_ElapsedTime += fElapsedTimeSeconds;
	}

	void Render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer = frame->getFramebuffer();
		auto fbInfo							 = framebuffer->getFramebufferInfo();

		if (!m_DrawTexture) {
			nvrhi::TextureDesc textureDesc;
			textureDesc.width			 = fbInfo.width;
			textureDesc.height			 = fbInfo.height;
			textureDesc.format			 = nvrhi::Format::RGBA32_FLOAT;
			textureDesc.debugName		 = "Draw Texture";
			textureDesc.initialState	 = nvrhi::ResourceStates::ShaderResource;
			textureDesc.keepInitialState = true;
			textureDesc.isUAV			 = true;
			textureDesc.sampleCount		 = 1;
			m_DrawTexture				 = m_CUFriend->createExternalTexture(textureDesc);
			m_CommandList->open();
			m_CommandList->setPermanentTextureState(m_DrawTexture,
													nvrhi::ResourceStates::ShaderResource);
			m_CommandList->close();
			GetDevice()->executeCommandList(m_CommandList);
		}
		
		//if (!m_Pipeline) {
		//	nvrhi::GraphicsPipelineDesc psoDesc;
		//	psoDesc.VS		 = m_VertexShader;
		//	psoDesc.PS		 = m_PixelShader;
		//	psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
		//	psoDesc.renderState.depthStencilState.depthTestEnable = false;
		//	m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		//}

		//m_CommandList->open();
		//nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
		//nvrhi::GraphicsState state;
		//state.pipeline	  = m_Pipeline;
		//state.framebuffer = framebuffer;
		//state.viewport.addViewportAndScissorRect(fbInfo.getViewport());
		//m_CommandList->setGraphicsState(state);

		//nvrhi::DrawArguments args;
		//args.vertexCount = 3;
		//m_CommandList->draw(args);
		//m_CommandList->close();
		//GetDevice()->executeCommandList(m_CommandList);
		GetDevice()->waitForIdle();

#if 0
		CUDA_SYNC_CHECK();
		cudaSurfaceObject_t cudaFrame = frame->getMappedCudaSurface(m_CUFriend);
		drawScreen(cudaFrame, m_ElapsedTime, fbInfo.width, fbInfo.height);
		CUDA_SYNC_CHECK();
#else
		static bool initialized{};
		
		if (initialized) {
			CUDA_SYNC_CHECK();
			if (!m_DrawCudaSurface)
				m_DrawCudaSurface =
					m_CUFriend->mapVulkanTextureToCudaSurface(m_DrawTexture, cudaArrayDefault);
			drawScreen(m_DrawCudaSurface, m_ElapsedTime, fbInfo.width, fbInfo.height);
			CUDA_SYNC_CHECK();
		} else {
			initialized = true;
		}
		
		m_CommandList->open();
		m_CommandList->setTextureState(m_DrawTexture, nvrhi::AllSubresources,
									   nvrhi::ResourceStates::ShaderResource);
		m_HelperPass->BlitTexture(m_CommandList, framebuffer, m_DrawTexture, m_BindingCache.get());
		m_CommandList->close();
		GetDevice()->executeCommandList(m_CommandList);
#endif
	}
};


extern "C" int main(int argc, const char *argv[]) {
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);
	
	DeviceCreationParameters deviceParams		= {};
	deviceParams.backBufferWidth				= 800;
	deviceParams.backBufferHeight				= 600;
	deviceParams.enableDebugRuntime				= true;
	deviceParams.enableNvrhiValidationLayer		= true;
	deviceParams.swapChainFormat				= nvrhi::Format::SRGBA8_UNORM;
	deviceParams.renderFormat					= nvrhi::Format::RGBA32_FLOAT;				
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