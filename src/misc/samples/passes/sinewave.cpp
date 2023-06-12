#include <common.h>
#include <logger.h>
#include <krrmath/clipspace.h>
#include <nvrhi/vulkan.h>
#include <util/check.h>
#include <renderpass.h>

#include "deviceprog.h"
#include "main/renderer.h"
#include "vulkan/textureloader.h"
#include "vulkan/shader.h"
#include "vulkan/cuvk.h"

KRR_NAMESPACE_BEGIN

using namespace vkrhi;
using namespace cuvk;

const char g_WindowTitle[] = "Sine Wave Simulator";

class WaveRenderer: public RenderPass {
public:
	using RenderPass::RenderPass;

	struct ConstantBufferEntry {
		Matrix4f mvp;
		float padding[16 * 3];
	};
	
	void initialize() {
		m_sim = SineWaveSimulation(256, 256);
		m_CuVkHandler = std::make_shared<CuVkHandler>(getVulkanDevice());
	
		// initialize CUDA
		int cuda_device = m_CuVkHandler->initCUDA();	// selected cuda device 
		if (cuda_device == -1) {
			Log(Error, "No CUDA-Vulkan interop capable device found\n");
			exit(EXIT_FAILURE);
		}
		m_sim.initCudaLaunchConfig(cuda_device);
		CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
		
		// creating the shader modules
		ShaderLoader shaderLoader(getVulkanDevice());
		m_vertexShader = shaderLoader.createShader(
			"src/misc/samples/passes/shaders/sinewave.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
		m_pixelShader = shaderLoader.createShader(
			"src/misc/samples/passes/shaders/sinewave.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

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
		m_inputLayout = getVulkanDevice()->createInputLayout(attributes, std::size(attributes), m_vertexShader);

		m_commandList = getVulkanDevice()->createCommandList();
		m_commandList->open();

		nvrhi::BufferDesc indexBufferDesc;
		indexBufferDesc.isIndexBuffer		= true;
		indexBufferDesc.byteSize			= sizeof(uint32_t) * (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;
		indexBufferDesc.debugName			= "IndexBuffer";
		indexBufferDesc.initialState		= nvrhi::ResourceStates::CopyDest;
		m_indexBuffer						= getVulkanDevice()->createBuffer(indexBufferDesc);
	
		nvrhi::BufferDesc xyBufferDesc;
		xyBufferDesc.isVertexBuffer = true;
		xyBufferDesc.byteSize		= sizeof(Vector2f) * m_sim.getWidth() * m_sim.getHeight();
		xyBufferDesc.debugName		= "XYBuffer";
		xyBufferDesc.initialState	= nvrhi::ResourceStates::CopyDest;
		m_xyBuffer					= getVulkanDevice()->createBuffer(xyBufferDesc);

		nvrhi::BufferDesc heightBufferDesc;
		heightBufferDesc.isVertexBuffer = true;
		heightBufferDesc.byteSize		= sizeof(float) * m_sim.getWidth() * m_sim.getHeight();
		heightBufferDesc.debugName		= "HeightBuffer";
		heightBufferDesc.initialState	= nvrhi::ResourceStates::CopyDest;
		m_heightBuffer = m_CuVkHandler->createExternalBuffer(heightBufferDesc);

		m_constantBuffer = getVulkanDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(
										  sizeof(ConstantBufferEntry), "ConstantBuffer")
				.setInitialState(nvrhi::ResourceStates::ConstantBuffer)
				.setKeepInitialState(true));

		m_commandList->beginTrackingBufferState(m_heightBuffer, nvrhi::ResourceStates::CopyDest);
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
		getVulkanDevice()->executeCommandList(m_commandList);

		m_CuVkHandler->importVulkanBufferToCudaPtr((void**)&m_cudaHeightData, m_cudaHeightMem, m_heightBuffer);
		m_sim.initSimulation(m_cudaHeightData);

		Log(Info, "Creating binding set and layout");
		nvrhi::BindingSetDesc bindingSetDesc;
		bindingSetDesc.bindings = {
			nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer, nvrhi::EntireBuffer)
		};
		if (!nvrhi::utils::CreateBindingSetAndLayout(getVulkanDevice(), nvrhi::ShaderType::All, 0,
													 bindingSetDesc, m_bindingLayout, m_bindingSet))
			logError("Failed to create the binding set and layout");
		Log(Info, "Finished simulator initialization");
	}

	void tick(float seconds) override {
		m_elapsedTime += seconds;
		m_sim.stepSimulation(m_elapsedTime, m_stream);
	}

	void resizing() override { m_pipeline = nullptr; }

	void render(RenderContext *context) override { 
		nvrhi::FramebufferHandle framebuffer = context->getFramebuffer();
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
			m_pipeline = getVulkanDevice()->createGraphicsPipeline(psoDesc, framebuffer);
			Log(Info, "Finished initializing pipeline");
		}
	
		m_commandList->open();

		nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.2f));
		
		Vector3f eye	= {1.75f, 1.75f, 1.25f};
		Vector3f center = {0.0f, 0.0f, -0.35};
		Vector3f up		= {0.0f, 0.0f, 1.0f};

		Matrix4f view = look_at(eye, center, up);
		Matrix4f proj = perspective(radians(45.f), float(fbInfo.width) / fbInfo.height, 0.1f, 10.f);
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
		getVulkanDevice()->executeCommandList(m_commandList);
	}

private:
	double m_elapsedTime{0};
	cudaStream_t m_stream{0};

	std::shared_ptr<CuVkHandler> m_CuVkHandler;
	SineWaveSimulation m_sim;
	float *m_cudaHeightData;
	cudaExternalMemory_t m_cudaHeightMem;
	
	nvrhi::ShaderHandle m_vertexShader, m_pixelShader; 
	nvrhi::BufferHandle m_heightBuffer, m_xyBuffer, m_indexBuffer, m_constantBuffer;
	nvrhi::InputLayoutHandle m_inputLayout;
	nvrhi::BindingLayoutHandle m_bindingLayout;
	nvrhi::BindingSetHandle m_bindingSet;
	nvrhi::GraphicsPipelineHandle m_pipeline{};
	nvrhi::CommandListHandle m_commandList;
};

extern "C" int main(int argc, const char *argv[]) {
	auto app = std::make_unique<RenderApp>();
	app->setWindowTitle(g_WindowTitle);
	app->addRenderPassToFront(std::make_shared<WaveRenderer>());
	app->run();
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END