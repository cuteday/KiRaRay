#include <nvrhi/utils.h>
#include <common.h>
#include <logger.h>
#include <renderpass.h>
#include <nvrhi/nvrhi.h>

#include "main/renderer.h"
#include "graphics/textureloader.h"
#include "graphics/shader.h"
#include "deviceprog.h"

NAMESPACE_BEGIN(krr)

static const char *g_WindowTitle = "HelloGraphicsCuda";

class HelloGraphicsCuda : public RenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;
	float m_ElapsedTime{};
	
public:
	using RenderPass::RenderPass;
	bool isCudaPass() const override { return false; }

	void initialize() override {
		ShaderLoader shaderLoader(getDevice());

		m_VertexShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_vs", nullptr,
													nvrhi::ShaderType::Vertex);
		m_PixelShader = shaderLoader.createShader("src/misc/samples/passes/shaders/triangle.hlsl", "main_ps", nullptr,
													nvrhi::ShaderType::Pixel);
		

		if (!m_VertexShader || !m_PixelShader) 
			Log(Fatal, "Shader initialization failed");
		

		m_CommandList = getDevice()->createCommandList();
	}

	void resizing() override { 
		m_Pipeline = nullptr; 
	}

	void tick(float fElapsedTimeSeconds) override {
		m_ElapsedTime = fElapsedTimeSeconds;
	}

	void render(RenderContext *context) override {
		nvrhi::FramebufferHandle framebuffer = context->getFramebuffer();
		auto fbInfo							 = framebuffer->getFramebufferInfo();
		
		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS		 = m_VertexShader;
			psoDesc.PS		 = m_PixelShader;
			psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;
			m_Pipeline = getDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

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
		getDevice()->executeCommandList(m_CommandList);

		RenderContext::CudaScope cudaScope(context);
		auto cudaRenderTarget = context->getColorTexture()->getCudaRenderTarget();
		drawScreen(context->getCudaStream(), cudaRenderTarget, m_ElapsedTime, fbInfo.width,
				   fbInfo.height);

	}
};

extern "C" int main(int argc, const char *argv[]) {
	auto app = std::make_unique<RenderApp>();
	app->setWindowTitle(g_WindowTitle);
	app->addRenderPassToFront(std::make_shared<HelloGraphicsCuda>());
	app->run();
	exit(EXIT_SUCCESS);
}

NAMESPACE_END(krr)