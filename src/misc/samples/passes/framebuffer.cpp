// https://www.khronos.org/blog/understanding-vulkan-synchronization
#include <common.h>
#include <logger.h>
#include <renderpass.h>
#include <nvrhi/vulkan.h>

#include "window.h"
#include "vulkan/textureloader.h"
#include "vulkan/shader.h"
#include "vulkan/cufriends.h"
#include "deviceprog.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "CuFramebuffer";

class HelloVkCuda : public RenderPass {
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
	using RenderPass::RenderPass;

	~HelloVkCuda() { 
		getVulkanNativeDevice().destroySemaphore(m_CudaUpdateVkSem);
		getVulkanNativeDevice().destroySemaphore(m_VkUpdateCudaSem);
	}

	void initialize() {
		ShaderLoader shaderLoader(getVulkanDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);
		
		m_CUFriend = std::make_shared<vkrhi::CudaVulkanFriend>(getVulkanDevice());

		if (!m_VertexShader || !m_PixelShader) 
			Log(Fatal, "Shader initialization failed");
		
		CUDA_CHECK(cudaStreamCreate(&m_CudaStream));

		m_CommandList = getVulkanDevice()->createCommandList();
		m_CUFriend->createExternalSemaphore(m_CudaUpdateVkSem);
		m_CUFriend->createExternalSemaphore(m_VkUpdateCudaSem);
		m_CudaExtCudaUpdateVkSem = m_CUFriend->importVulkanSemaphoreToCuda(m_CudaUpdateVkSem);
		m_CudaExtVkUpdateCudaSem = m_CUFriend->importVulkanSemaphoreToCuda(m_VkUpdateCudaSem);
	}

	void resizing() override { 
		m_Pipeline = nullptr; 
	}

	void tick(float fElapsedTimeSeconds) override {
		m_ElapsedTime += fElapsedTimeSeconds;
	}

	void render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer = frame->getFramebuffer();
		auto fbInfo							 = framebuffer->getFramebufferInfo();
		
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;
			m_Pipeline = getVulkanDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

		getVulkanDevice()->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, m_VkUpdateCudaSem, 0);
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
		getVulkanDevice()->executeCommandList(m_CommandList);

		vkrhi::CudaVulkanFriend::cudaWaitExternalSemaphore(m_CudaStream, 0,
														   &m_CudaExtVkUpdateCudaSem);
		auto cudaRenderTarget		  = frame->getCudaRenderTarget();
		drawScreen(m_CudaStream, cudaRenderTarget, m_ElapsedTime, fbInfo.width,
				   fbInfo.height);
		CUDA_SYNC_CHECK();

	}
};

extern "C" int main(int argc, const char *argv[]) {
	auto app							  = std::make_unique<DeviceManager>();
	DeviceCreationParameters deviceParams = {};
	deviceParams.backBufferWidth				= 800;
	deviceParams.backBufferHeight				= 600;
	deviceParams.enableDebugRuntime				= true;
	deviceParams.enableNvrhiValidationLayer		= true;
	deviceParams.swapChainFormat				= nvrhi::Format::SRGBA8_UNORM;
	deviceParams.renderFormat					= nvrhi::Format::RGBA32_FLOAT;				
	deviceParams.enableCudaInterop				= true;

	if (!app->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}
	{
		auto example = std::make_shared<HelloVkCuda>(app.get());
		example->initialize();
		app->AddRenderPassToBack(example);
		app->RunMessageLoop();
		app->RemoveRenderPass(example);
	}
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END