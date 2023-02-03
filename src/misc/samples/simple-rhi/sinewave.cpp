#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include "SineWaveSimulation.h"
#include "devicemanager.h"
#include "shader.h"

KRR_NAMESPACE_BEGIN

class WaveRenderer: public IRenderPass {
public:
	using IRenderPass::IRenderPass;
	
	void Initialize() {
		m_sim = SineWaveSimulation(512, 512);
	
		// creating the shader modules
		ShaderLoader shaderLoader(GetDevice());
		nvrhi::ShaderHandle vertexShader = shaderLoader.createShader(
			"src/misc/shader.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
		nvrhi::ShaderHandle fragShader = shaderLoader.createShader(
			"src/misc/shader.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
		
		nvrhi::BufferDesc indexBufferDesc;
		indexBufferDesc.isIndexBuffer		= true;
		indexBufferDesc.byteSize			= sizeof(uint32_t) * (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;
		indexBufferDesc.debugName			= "IndexBuffer";
		indexBufferDesc.initialState		= nvrhi::ResourceStates::CopyDest;
		indexBufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::None;
		m_indexBuffer						= GetDevice()->createBuffer(indexBufferDesc);
	
		nvrhi::BufferDesc xyBufferDesc;
		indexBufferDesc.byteSize = sizeof(Vector2f) * m_sim.getWidth() * m_sim.getHeight();
		indexBufferDesc.debugName			= "XYBuffer";
		indexBufferDesc.initialState		= nvrhi::ResourceStates::CopyDest;
		m_xyBuffer							= GetDevice()->createBuffer(indexBufferDesc);

		// upload permanent data to buffers
		m_commandList = GetDevice()->createCommandList();
		{
			Vector2f *vertices = static_cast<Vector2f *>(
				GetDevice()->mapBuffer(m_xyBuffer, nvrhi::CpuAccessMode::Write));
			for (size_t y = 0; y < m_sim.getHeight(); y++) {
				for (size_t x = 0; x < m_sim.getWidth(); x++) {
					vertices[y * m_sim.getWidth() + x][0] =
						(2.0f * x) / (m_sim.getWidth() - 1) - 1;
					vertices[y * m_sim.getWidth() + x][1] =
						(2.0f * y) / (m_sim.getHeight() - 1) - 1;
				}
			}
			GetDevice()->unmapBuffer(m_xyBuffer);
		}
		{
			uint32_t *indices = static_cast<uint32_t *>(
				GetDevice()->mapBuffer(m_indexBuffer, nvrhi::CpuAccessMode::Write));
			for (size_t y = 0; y < m_sim.getHeight() - 1; y++) {
				for (size_t x = 0; x < m_sim.getWidth() - 1; x++) {
					indices[0] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 0));
					indices[1] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 0));
					indices[2] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 1));
					indices[3] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 0));
					indices[4] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 1));
					indices[5] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 1));
					indices += 6;
				}
			}
			GetDevice()->unmapBuffer(m_indexBuffer);
		}

		m_commandList->open();
		m_commandList->setPermanentBufferState(m_indexBuffer, nvrhi::ResourceStates::IndexBuffer);
		//m_commandList->setPermanentBufferState(m_xyBuffer, nvrhi::ResourceStates::VertexBuffer);
		
		// creating the pipeline
		auto pipelineDesc = nvrhi::GraphicsPipelineDesc()
								.setVertexShader(vertexShader)
								.setFragmentShader(fragShader);

		// create binding set and layout
		nvrhi::BindingSetDesc bindingSetDesc;
		bindingSetDesc.bindings = {
			//nvrhi::BindingSetItem::ConstantBuffer(0, m_)
		};
	}

	void Animate(float seconds) override {

	}

	void BackBufferResizing() override {

	}

	void Render(nvrhi::IFramebuffer* framebuffer) override {

	}

private:
	SineWaveSimulation m_sim;
	nvrhi::ShaderHandle m_vertexShader, m_pixelShader;
	nvrhi::BufferHandle m_heightBuffer, m_xyBuffer, m_indexBuffer;
	nvrhi::InputLayoutHandle m_inputLayout;
	nvrhi::BindingLayoutHandle m_bindingLayout;
	nvrhi::BindingSetHandle m_bindingSet;
	nvrhi::GraphicsPipelineHandle m_pipeline;
	nvrhi::CommandListHandle m_commandList;
};

extern "C" int main(int argc, const char *argv[]) {
	Log(Info, "Hello from simple-rhi!");
	
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);

	DeviceCreationParameters deviceParams	= {};
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, "Sine Wave")) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}

	{ 
		WaveRenderer app(deviceManager);
		app.Initialize();

		deviceManager->AddRenderPassToBack(&app);
		deviceManager->RunMessageLoop();
		deviceManager->RemoveRenderPass(&app);
	}

	deviceManager->Shutdown();
	delete deviceManager;
	
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END