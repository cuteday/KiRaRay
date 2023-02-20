#include <common.h>
#include <logger.h>
#include <krrmath/transform.h>
#include <nvrhi/vulkan.h>
#include <util/check.h>

#include "common/helperpass.h"
#include "common/devicemanager.h"
#include "common/textureloader.h"
#include "common/shader.h"

KRR_NAMESPACE_BEGIN

static const char *g_WindowTitle = "Box-Shii-s";

struct Vertex {
	Vector3f position;
	Vector2f uv;
};

static const Vertex g_Vertices[] = {
	{{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}}, // front face
	{{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}},  {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}},
	{{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},

	{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}}, // right side face
	{{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},	   {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},
	{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},

	{{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}}, // left side face
	{{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}}, {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
	{{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},

	{{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}}, // back face
	{{-0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},  {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},

	{{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f}}, // top face
	{{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},	   {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},

	{{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}}, // bottom face
	{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}}, {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
	{{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
};

static const uint32_t g_Indices[] = {
	0,	1,	2,	0,	3,	1,	// front face
	4,	5,	6,	4,	7,	5,	// left face
	8,	9,	10, 8,	11, 9,	// right face
	12, 13, 14, 12, 15, 13, // back face
	16, 17, 18, 16, 19, 17, // top face
	20, 21, 22, 20, 23, 21, // bottom face
};

constexpr uint32_t c_NumViews = 4;

static const Vector3f g_RotationAxes[c_NumViews] = {
	Vector3f(1.f, 0.5f, 0.f),
	Vector3f(0.f, 1.f, 0.5f),
	Vector3f(0.5f, 0.f, 1.f),
	Vector3f(1.f, 1.f, 1.f),
};

class VertexBuffer : public IRenderPass {
private:
	nvrhi::ShaderHandle m_VertexShader;
	nvrhi::ShaderHandle m_PixelShader;
	nvrhi::BufferHandle m_ConstantBuffer;
	nvrhi::BufferHandle m_VertexBuffer;
	nvrhi::BufferHandle m_IndexBuffer;
	nvrhi::TextureHandle m_Texture;
	nvrhi::InputLayoutHandle m_InputLayout;
	nvrhi::BindingLayoutHandle m_BindingLayout;
	nvrhi::BindingSetHandle m_BindingSets[c_NumViews];
	nvrhi::GraphicsPipelineHandle m_Pipeline;
	nvrhi::CommandListHandle m_CommandList;
	float m_Rotation = 0.f;

public:
	using IRenderPass::IRenderPass;

	// This example uses a single large constant buffer with multiple views to draw multiple
	// versions of the same model. The alignment and size of partially bound constant buffers must
	// be a multiple of 256 bytes, so define a struct that represents one constant buffer entry or
	// slice for one draw call.
	struct ConstantBufferEntry {
		Matrix4f viewProjMatrix;
		float padding[16 * 3];
	};

	static_assert(sizeof(ConstantBufferEntry) == nvrhi::c_ConstantBufferOffsetSizeAlignment,
				  "sizeof(ConstantBufferEntry) must be 256 bytes");

	bool Init() {
		std::shared_ptr<ShaderLoader> shaderLoader = std::make_shared<ShaderLoader>(GetDevice());
		m_VertexShader = shaderLoader->createShader("src/misc/samples/simple-rhi/shaders/box.hlsl",
													"main_vs", nullptr,
													 nvrhi::ShaderType::Vertex);
		m_PixelShader  = shaderLoader->createShader("src/misc/samples/simple-rhi/shaders/box.hlsl",
													"main_ps", nullptr,
													 nvrhi::ShaderType::Pixel);

		if (!m_VertexShader || !m_PixelShader) {
			return false;
		}

		m_ConstantBuffer = GetDevice()->createBuffer(
			nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(ConstantBufferEntry) * c_NumViews,
														 "ConstantBuffer")
				.setInitialState(nvrhi::ResourceStates::ConstantBuffer)
				.setKeepInitialState(true));

		nvrhi::VertexAttributeDesc attributes[] = {
			nvrhi::VertexAttributeDesc()
				.setName("POSITION")
				.setFormat(nvrhi::Format::RGB32_FLOAT)
				.setOffset(offsetof(Vertex, position))
				.setElementStride(sizeof(Vertex)),
			nvrhi::VertexAttributeDesc()
				.setName("UV")
				.setFormat(nvrhi::Format::RG32_FLOAT)
				.setOffset(offsetof(Vertex, uv))
				.setElementStride(sizeof(Vertex)),
		};
		m_InputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)),
													   m_VertexShader);

		CommonRenderPasses commonPasses(GetDevice(), shaderLoader);
		TextureCache textureCache(GetDevice(), nullptr);

		m_CommandList = GetDevice()->createCommandList();
		m_CommandList->open();

		nvrhi::BufferDesc vertexBufferDesc;
		vertexBufferDesc.byteSize		= sizeof(g_Vertices);
		vertexBufferDesc.isVertexBuffer = true;
		vertexBufferDesc.debugName		= "VertexBuffer";
		vertexBufferDesc.initialState	= nvrhi::ResourceStates::CopyDest;
		m_VertexBuffer					= GetDevice()->createBuffer(vertexBufferDesc);

		m_CommandList->beginTrackingBufferState(m_VertexBuffer, nvrhi::ResourceStates::CopyDest);
		m_CommandList->writeBuffer(m_VertexBuffer, g_Vertices, sizeof(g_Vertices));
		m_CommandList->setPermanentBufferState(m_VertexBuffer, nvrhi::ResourceStates::VertexBuffer);

		nvrhi::BufferDesc indexBufferDesc;
		indexBufferDesc.byteSize	  = sizeof(g_Indices);
		indexBufferDesc.isIndexBuffer = true;
		indexBufferDesc.debugName	  = "IndexBuffer";
		indexBufferDesc.initialState  = nvrhi::ResourceStates::CopyDest;
		m_IndexBuffer				  = GetDevice()->createBuffer(indexBufferDesc);

		m_CommandList->beginTrackingBufferState(m_IndexBuffer, nvrhi::ResourceStates::CopyDest);
		m_CommandList->writeBuffer(m_IndexBuffer, g_Indices, sizeof(g_Indices));
		m_CommandList->setPermanentBufferState(m_IndexBuffer, nvrhi::ResourceStates::IndexBuffer);

		std::filesystem::path textureFileName =
			"src/misc/samples/simple-rhi/assets/emoticon_001.png";
		std::shared_ptr<LoadedTexture> texture =
			textureCache.LoadTextureFromFile(textureFileName, true, nullptr, m_CommandList);
		m_Texture = texture->texture;

		m_CommandList->close();
		GetDevice()->executeCommandList(m_CommandList);

		if (!texture->texture) {
			logError("Couldn't load the texture");
			return false;
		}

		// Create a single binding layout and multiple binding sets, one set per view.
		// The different binding sets use different slices of the same constant buffer.
		for (uint32_t viewIndex = 0; viewIndex < c_NumViews; ++viewIndex) {
			nvrhi::BindingSetDesc bindingSetDesc;
			bindingSetDesc.bindings = {
				// Note: using viewIndex to construct a buffer range.
				nvrhi::BindingSetItem::ConstantBuffer(
					0, m_ConstantBuffer,
					nvrhi::BufferRange(sizeof(ConstantBufferEntry) * viewIndex,
									   sizeof(ConstantBufferEntry))),
				// Texutre and sampler are the same for all model views.
				nvrhi::BindingSetItem::Texture_SRV(0, m_Texture),
				nvrhi::BindingSetItem::Sampler(0, commonPasses.m_AnisotropicWrapSampler)};

			// Create the binding layout (if it's empty -- so, on the first iteration) and the
			// binding set.
			if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
														 bindingSetDesc, m_BindingLayout,
														 m_BindingSets[viewIndex])) {
				logError("Couldn't create the binding set or layout");
				return false;
			}
		}

		return true;
	}

	void Animate(float seconds) override {
		m_Rotation += seconds * 1.1f;
		GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
	}

	void BackBufferResizing() override { m_Pipeline = nullptr; }

	void Render(RenderFrame::SharedPtr frame) override {
		nvrhi::FramebufferHandle framebuffer   = frame->getFramebuffer();
		const nvrhi::FramebufferInfoEx &fbinfo = framebuffer->getFramebufferInfo();

		if (!m_Pipeline) {
			nvrhi::GraphicsPipelineDesc psoDesc;
			psoDesc.VS			   = m_VertexShader;
			psoDesc.PS			   = m_PixelShader;
			psoDesc.inputLayout	   = m_InputLayout;
			psoDesc.bindingLayouts = {m_BindingLayout};
			psoDesc.primType	   = nvrhi::PrimitiveType::TriangleList;
			psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::Back;
			psoDesc.renderState.rasterState.frontCounterClockwise = true;
			psoDesc.renderState.depthStencilState.depthTestEnable = false;

			m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
		}

		m_CommandList->open();

		nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.2f));

		ConstantBufferEntry modelConstants[c_NumViews];
		for (uint32_t viewIndex = 0; viewIndex < c_NumViews; ++viewIndex) {
			Matrix4f modelMatrix = Matrix4f::Identity();
			modelMatrix.topLeftCorner<3, 3>() =
				Eigen::AngleAxis(m_Rotation, normalize(g_RotationAxes[viewIndex])).matrix();
			Matrix4f viewMatrix = look_at(Vector3f{0, -1, 3.5}, {0, 0, 0}, {0, -1, 0}).matrix();
			Matrix4f projMatrix = perspective(
				radians(30.f), float(fbinfo.width) / float(fbinfo.height), 0.01f, 1000.f);
			Matrix4f mvp = projMatrix * viewMatrix * modelMatrix;
			//std::cout << "Proj: \n" << projMatrix << "\n";
			//std::cout << "View: \n" << viewMatrix << "\n";
			modelConstants[viewIndex].viewProjMatrix = mvp;
		}

		m_CommandList->writeBuffer(m_ConstantBuffer, modelConstants, sizeof(modelConstants));

		for (uint32_t viewIndex = 0; viewIndex < c_NumViews; ++viewIndex) {
			nvrhi::GraphicsState state;
			state.bindings		= {m_BindingSets[viewIndex]};
			state.indexBuffer	= {m_IndexBuffer, nvrhi::Format::R32_UINT, 0};
			state.vertexBuffers = {{m_VertexBuffer, 0, 0}};
			state.pipeline		= m_Pipeline;
			state.framebuffer	= framebuffer;

			const float width  = float(fbinfo.width) * 0.5f;
			const float height = float(fbinfo.height) * 0.5f;
			const float left   = width * float(viewIndex % 2);
			const float top	   = height * float(viewIndex / 2);

			const nvrhi::Viewport viewport =
				nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);
			state.viewport.addViewportAndScissorRect(viewport);

			m_CommandList->setGraphicsState(state);
			
			nvrhi::DrawArguments args;
			args.vertexCount = std::size(g_Indices);
			m_CommandList->drawIndexed(args);
		}

		m_CommandList->close();
		GetDevice()->executeCommandList(m_CommandList);
	}
};


 extern "C" int main(int argc, const char *argv[]) {
	DeviceManagerImpl *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);

	DeviceCreationParameters deviceParams;
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}

	{
		VertexBuffer example(deviceManager);
		if (example.Init()) {
			deviceManager->AddRenderPassToBack(&example);
			deviceManager->RunMessageLoop();
			deviceManager->RemoveRenderPass(&example);
		}
	}

	deviceManager->Shutdown();
	delete deviceManager;
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END