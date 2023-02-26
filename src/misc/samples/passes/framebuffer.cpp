// https://www.khronos.org/blog/understanding-vulkan-synchronization
#include <common.h>
#include <logger.h>
#include <renderpass.h>
#include <nvrhi/vulkan.h>

#include "main/renderer.h"
#include "vulkan/textureloader.h"
#include "vulkan/shader.h"
#include "vulkan/cuvk.h"
#include "deviceprog.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "HelloVkCuda";

class HelloVkCuda : public RenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;
	std::shared_ptr<vkrhi::CuVkHandler> m_CuVkHandler;
	float m_ElapsedTime{};
	CUstream m_CudaStream{};
	vkrhi::CuVkSemaphore m_CudaUpdateVkSem, m_VkUpdateCudaSem;
	
public:
	using RenderPass::RenderPass;

	~HelloVkCuda() { 
		getVulkanNativeDevice().destroySemaphore(m_CudaUpdateVkSem);
		getVulkanNativeDevice().destroySemaphore(m_VkUpdateCudaSem);
	}

	void initialize() override {
		ShaderLoader shaderLoader(getVulkanDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);
		
		m_CuVkHandler = std::make_shared<vkrhi::CuVkHandler>(getVulkanDevice());

		if (!m_VertexShader || !m_PixelShader) 
			Log(Fatal, "Shader initialization failed");
		
		CUDA_CHECK(cudaStreamCreate(&m_CudaStream));

		m_CommandList = getVulkanDevice()->createCommandList();
		m_CudaUpdateVkSem = m_CuVkHandler->createCuVkSemaphore();
		m_VkUpdateCudaSem = m_CuVkHandler->createCuVkSemaphore();
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

		vkrhi::CuVkHandler::cudaWaitExternalSemaphore(m_CudaStream, 0,
														   &m_VkUpdateCudaSem.cuda());
		auto cudaRenderTarget		  = frame->getCudaRenderTarget();
		drawScreen(m_CudaStream, cudaRenderTarget, m_ElapsedTime, fbInfo.width,
				   fbInfo.height);
		CUDA_SYNC_CHECK();

	}
};

extern "C" int main(int argc, const char *argv[]) {
	auto app = std::make_unique<RenderApp>();
	app->SetWindowTitle(g_WindowTitle);
	app->AddRenderPassToFront(std::make_shared<HelloVkCuda>());
	app->run();
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END