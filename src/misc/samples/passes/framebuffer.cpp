// https://www.khronos.org/blog/understanding-vulkan-synchronization
#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include "common/devicemanager.h"
#include "vulkan/textureloader.h"
#include "vulkan/shader.h"
#include "vulkan/cufriends.h"
#include "deviceprog.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "CuFramebuffer";

class HelloVkCuda : public IRenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;
	std::shared_ptr<vkrhi::CudaVulkanFriend> m_CUFriend;
	float m_ElapsedTime{};
	CUstream m_CudaStream{};
	vk::Semaphore m_CudaUpdateVkSem, m_VkUpdateCudaSem;
	cudaExternalSemaphore_t m_CudaExtCudaUpdateVkSem, m_CudaExtVkUpdateCudaSem;

public:
	using IRenderPass::IRenderPass;

	~HelloVkCuda() { 
		GetNativeDevice().destroySemaphore(m_CudaUpdateVkSem);
		GetNativeDevice().destroySemaphore(m_VkUpdateCudaSem);
	}

	bool Init() {
		ShaderLoader shaderLoader(GetDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);
		
		m_CUFriend = std::make_shared<vkrhi::CudaVulkanFriend>(GetDevice(false));

		if (!m_VertexShader || !m_PixelShader) 
			return false;
		
		CUDA_CHECK(cudaStreamCreate(&m_CudaStream));

		m_CommandList = GetDevice()->createCommandList();
		m_CUFriend->createExternalSemaphore(m_CudaUpdateVkSem);
		m_CUFriend->createExternalSemaphore(m_VkUpdateCudaSem);
		m_CudaExtCudaUpdateVkSem = m_CUFriend->importVulkanSemaphoreToCuda(m_CudaUpdateVkSem);
		m_CudaExtVkUpdateCudaSem = m_CUFriend->importVulkanSemaphoreToCuda(m_VkUpdateCudaSem);
		return true;
	}

	void BackBufferResizing() override { 
		m_Pipeline = nullptr; 
	}

	void Animate(float fElapsedTimeSeconds) override {
		GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
		m_ElapsedTime += fElapsedTimeSeconds;
	}

	void Render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer = frame->getFramebuffer();
		auto fbInfo							 = framebuffer->getFramebufferInfo();
		
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;
			m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

		GetVkDevice()->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, m_VkUpdateCudaSem, 0);
		m_CommandList->open();
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
		GetDevice()->executeCommandList(m_CommandList);

		vkrhi::CudaVulkanFriend::cudaWaitExternalSemaphore(m_CudaStream, 0,
														   &m_CudaExtVkUpdateCudaSem);
		cudaSurfaceObject_t cudaFrame = frame->getMappedCudaSurface(m_CUFriend);
		drawScreen(m_CudaStream, cudaFrame, m_ElapsedTime, fbInfo.width, fbInfo.height);
		CUDA_SYNC_CHECK();

	}
};


extern "C" int main(int argc, const char *argv[]) {
	DeviceManagerImpl *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);
	
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
		HelloVkCuda example(deviceManager);
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