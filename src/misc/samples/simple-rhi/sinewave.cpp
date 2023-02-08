#include <common.h>
#include <logger.h>
#include <krrmath/transform.h>
#include <nvrhi/vulkan.h>
#include <util/check.h>

#include "SineWaveSimulation.h"
#include "common/devicemanager.h"
#include "common/textureloader.h"
#include "common/shader.h"
#include "common/cufriends.h"

KRR_NAMESPACE_BEGIN

const char g_windowTitle[] = "Sine Wave Simulator";

class WaveRenderer: public IRenderPass {
public:
	using IRenderPass::IRenderPass;

	struct ConstantBufferEntry {
		Matrix4f mvp;
		float padding[16 * 3];
	};
	
	void Initialize() {
		m_sim = SineWaveSimulation(32, 32);
	
		// creating the shader modules
		ShaderLoader shaderLoader(GetDevice());
		m_vertexShader = shaderLoader.createShader(
			"src/misc/samples/simple-rhi/shaders/sinewave.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
		m_pixelShader = shaderLoader.createShader(
			"src/misc/samples/simple-rhi/shaders/sinewave.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

		if (!m_vertexShader || !m_pixelShader) {
			Log(Error, "Failed to create vertex and pixel shader");
			return;
		}

		nvrhi::VertexAttributeDesc attributes[] = {
			nvrhi::VertexAttributeDesc()
				.setName("POSITION")
				.setFormat(nvrhi::Format::RG32_FLOAT)
				.setOffset(0)
				.setElementStride(sizeof(Vector2f))
				.setBufferIndex(0),
			nvrhi::VertexAttributeDesc()
				.setName("HEIGHT")
				.setFormat(nvrhi::Format::R32_FLOAT)
				.setOffset(0)
				.setElementStride(sizeof(float))
				.setBufferIndex(1),
		};
		m_inputLayout = GetDevice()->createInputLayout(attributes, std::size(attributes), m_vertexShader);

		m_commandList = GetDevice()->createCommandList();
		m_commandList->open();

		nvrhi::BufferDesc indexBufferDesc;
		indexBufferDesc.isIndexBuffer		= true;
		indexBufferDesc.byteSize			= sizeof(uint32_t) * (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;
		indexBufferDesc.debugName			= "IndexBuffer";
		indexBufferDesc.initialState		= nvrhi::ResourceStates::CopyDest;
		m_indexBuffer						= GetDevice()->createBuffer(indexBufferDesc);
	
		nvrhi::BufferDesc xyBufferDesc;
		xyBufferDesc.isVertexBuffer = true;
		xyBufferDesc.byteSize		= sizeof(Vector2f) * m_sim.getWidth() * m_sim.getHeight();
		xyBufferDesc.debugName		= "XYBuffer";
		xyBufferDesc.initialState	= nvrhi::ResourceStates::CopyDest;
		m_xyBuffer					= GetDevice()->createBuffer(xyBufferDesc);

		nvrhi::BufferDesc heightBufferDesc;
		heightBufferDesc.isVertexBuffer = true;
		heightBufferDesc.byteSize		= sizeof(float) * m_sim.getWidth() * m_sim.getHeight();
		heightBufferDesc.debugName		= "HeightBuffer";
		heightBufferDesc.initialState	= nvrhi::ResourceStates::VertexBuffer;
		m_heightBuffer					= GetDevice()->createBuffer(heightBufferDesc);

		VkDevice device = GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_Device);
		
		m_constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(
										  sizeof(ConstantBufferEntry), "ConstantBuffer")
				.setInitialState(nvrhi::ResourceStates::ConstantBuffer)
				.setKeepInitialState(true));

		m_commandList->beginTrackingBufferState(m_heightBuffer, nvrhi::ResourceStates::VertexBuffer);
		m_commandList->beginTrackingBufferState(m_indexBuffer, nvrhi::ResourceStates::CopyDest);
		m_commandList->beginTrackingBufferState(m_xyBuffer, nvrhi::ResourceStates::CopyDest);
		
		// upload permanent data to buffers
		{
			Log(Info, "Mapping and writting xy buffer");
			static std::vector<Vector2f> vertices(m_sim.getWidth() * m_sim.getHeight());
						for (size_t y = 0; y < m_sim.getHeight(); y++) {
				for (size_t x = 0; x < m_sim.getWidth(); x++) {
					vertices[y * m_sim.getWidth() + x][0] = (2.0f * x) / (m_sim.getWidth() - 1) - 1;
					vertices[y * m_sim.getWidth() + x][1] =
						(2.0f * y) / (m_sim.getHeight() - 1) - 1;
				}
			}
			m_commandList->writeBuffer(m_xyBuffer, vertices.data(), sizeof(Vector2f) * vertices.size());
		}
		{
			Log(Info, "Mapping and writting index buffer");
			static std::vector<uint32_t> indices((m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6);
						for (size_t y = 0, base_ptr = 0; y < m_sim.getHeight() - 1; y++) {
				for (size_t x = 0; x < m_sim.getWidth() - 1; x++) {
					indices[base_ptr + 0] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 0));
					indices[base_ptr + 1] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 0));
					indices[base_ptr + 2] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 1));
					indices[base_ptr + 3] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 0));
					indices[base_ptr + 4] = (uint32_t) ((y + 1) * m_sim.getWidth() + (x + 1));
					indices[base_ptr + 5] = (uint32_t) ((y + 0) * m_sim.getWidth() + (x + 1));
					base_ptr += 6;
				}
			}
			m_commandList->writeBuffer(m_indexBuffer, indices.data(), sizeof(uint32_t) * indices.size());
		}

		Log(Info, "Specifying state for the buffers");		
		m_commandList->setPermanentBufferState(m_indexBuffer, nvrhi::ResourceStates::IndexBuffer);
		m_commandList->setPermanentBufferState(m_xyBuffer, nvrhi::ResourceStates::VertexBuffer);
		m_commandList->setPermanentBufferState(m_heightBuffer, nvrhi::ResourceStates::VertexBuffer);
		
		m_commandList->close();
		GetDevice()->executeCommandList(m_commandList);

		Log(Info, "Creating binding set and layout");
		nvrhi::BindingSetDesc bindingSetDesc;
		bindingSetDesc.bindings = {
			nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer, nvrhi::EntireBuffer)
		};
		if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
													 bindingSetDesc, m_bindingLayout, m_bindingSet))
			logError("Failed to create the binding set and layout");
		Log(Info, "Finished simulator initialization");
	}

	void Animate(float seconds) override {
		m_elapsedTime += seconds;
		//m_sim.stepSimulation(m_elapsedTime * 1e9, m_stream);
		GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle);
	}

	void BackBufferResizing() override { m_pipeline = nullptr; }

	void Render(nvrhi::IFramebuffer *framebuffer) override { 
		CUDA_SYNC_CHECK();

		auto &fbInfo = framebuffer->getFramebufferInfo();

		if (!m_pipeline) {
			Log(Info, "Initializing vulkan graphics pipeline");
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS			   = m_vertexShader;
			psoDesc.PS			   = m_pixelShader;
			psoDesc.inputLayout	   = m_inputLayout;
			psoDesc.bindingLayouts = {m_bindingLayout};
			psoDesc.primType	   = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;
			psoDesc.renderState.rasterState.fillMode = nvrhi::RasterFillMode::Wireframe;
			m_pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
			Log(Info, "Finished initializing pipeline");
		}
	
		m_commandList->open();

		nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.2f));
		
		Vector3f eye	= {1.75f, 1.75f, 1.25f};
		Vector3f center = {0.0f, 0.0f, -0.25f};
		Vector3f up		= {0.0f, 0.0f, 1.0f};

		Matrix4f view = look_at(eye, center, up);
		Matrix4f proj = perspective(radians(45.f), float(fbInfo.width) / fbInfo.height, 0.1f, 10.f);
		//proj(1, 1) *= -1.0f; // Flip y axis
		ConstantBufferEntry cb;
		cb.mvp = proj * view;
		m_commandList->writeBuffer(m_constantBuffer, &cb, sizeof(ConstantBufferEntry));
		
		nvrhi::GraphicsState state;
		state.indexBuffer	= {m_indexBuffer, nvrhi::Format::R32_UINT, 0};
		state.vertexBuffers = {{m_xyBuffer, 0, 0}, {m_heightBuffer, 1, 0},}; 
		state.bindings		= {m_bindingSet};
		state.pipeline		= m_pipeline;
		state.framebuffer	= framebuffer;
		state.viewport.addViewportAndScissorRect(fbInfo.getViewport());

		m_commandList->setGraphicsState(state);

		nvrhi::DrawArguments args;
		args.vertexCount = (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;
		m_commandList->drawIndexed(args);
		
		m_commandList->close();
		GetDevice()->executeCommandList(m_commandList);
	}
	
protected:

	

private:
	double m_elapsedTime{0};
	cudaStream_t m_stream{0};

	SineWaveSimulation m_sim;
	nvrhi::ShaderHandle m_vertexShader, m_pixelShader;
	nvrhi::BufferHandle m_heightBuffer, m_xyBuffer, m_indexBuffer, m_constantBuffer;
	nvrhi::InputLayoutHandle m_inputLayout;
	nvrhi::BindingLayoutHandle m_bindingLayout;
	nvrhi::BindingSetHandle m_bindingSet;
	nvrhi::GraphicsPipelineHandle m_pipeline{};
	nvrhi::CommandListHandle m_commandList;
};

extern "C" int main(int argc, const char *argv[]) {
	Log(Info, "Hello from simple-rhi!");
	
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);

	DeviceCreationParameters deviceParams	= {};
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle)) {
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