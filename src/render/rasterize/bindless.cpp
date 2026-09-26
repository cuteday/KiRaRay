#include "bindless.h"
#include <nvrhi/utils.h>
#include "graphics/ui.h"
#include "graphics/device.h"
#include "graphics/scene.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

class GBufferRenderTargets {
protected:
	Vector2i mSize{};
	uint mSampleCount{};

public:
	nvrhi::TextureHandle depth;
	nvrhi::TextureHandle diffuse;
	nvrhi::TextureHandle specular;
	nvrhi::TextureHandle normals;
	nvrhi::TextureHandle emissive;

	virtual bool isUpdateNeeded(Vector2i size, uint sampleCount) {
		return mSize != size || mSampleCount != sampleCount;
	}

	virtual void initialize(nvrhi::IDevice* device,
		Vector2i size,
		uint sampleCount) {
		nvrhi::TextureDesc desc;
		desc.width			  = size[0];
		desc.height			  = size[1];
		desc.initialState	  = nvrhi::ResourceStates::DepthWrite;
		desc.keepInitialState = true;
		desc.isRenderTarget	  = true;
		desc.useClearValue	  = true;
		desc.isTypeless		  = true;
		desc.isUAV			  = false;
		desc.sampleCount	  = sampleCount;
		desc.mipLevels		  = 1;
		desc.format			  = nvrhi::Format::D24S8;
		desc.clearValue		  = nvrhi::Color(1.f, 0.f, 0.f, 0.f);
		desc.debugName		  = "DepthBuffer";
		desc.dimension = sampleCount > 1 ? nvrhi::TextureDimension::Texture2DMS
										 : nvrhi::TextureDimension::Texture2D;
		depth				= device->createTexture(desc);
	
		mSize = size;
		mSampleCount = sampleCount;
	}

	virtual void clear(nvrhi::ICommandList* commandList) {
		const nvrhi::FormatInfo depthFormatInfo = nvrhi::getFormatInfo(depth->getDesc().format);
		commandList->clearDepthStencilTexture(depth, nvrhi::AllSubresources,
			true, 1.f, depthFormatInfo.hasStencil, 0);
	}
};

class RenderTargets : public GBufferRenderTargets {
public:
	nvrhi::TextureHandle color;		// potentially a MSAA texture

	virtual void initialize(nvrhi::IDevice *device, Vector2i size,
							uint sampleCount) override {
		GBufferRenderTargets::initialize(device, size, sampleCount);

		nvrhi::TextureDesc desc;
		desc.width			  = size[0];
		desc.height			  = size[1];
		desc.initialState	  = nvrhi::ResourceStates::RenderTarget;
		desc.keepInitialState = true;
		desc.isRenderTarget	  = true;
		desc.useClearValue	  = true;
		desc.sampleCount	  = sampleCount;
		desc.mipLevels		  = 1;
		desc.format			  = nvrhi::Format::RGBA32_FLOAT;
		desc.clearValue		  = nvrhi::Color(0.f);
		desc.debugName		  = "ColorBuffer";
		desc.dimension = sampleCount > 1 ? nvrhi::TextureDimension::Texture2DMS
										 : nvrhi::TextureDimension::Texture2D;
		color		   = device->createTexture(desc);
	}

	virtual void clear(nvrhi::ICommandList *commandList) {
		GBufferRenderTargets::clear(commandList);
		commandList->clearTextureFloat(color, nvrhi::AllSubresources,
											  nvrhi::Color(0.f));
	}
};

void BindlessRender::initialize() {
	mShaderLoader = std::make_shared<ShaderLoader>(getDevice());
	mBindingCache = std::make_shared<BindingCache>(getDevice());
	mHelperPass	  = std::make_shared<CommonRenderPasses>(getDevice(), mShaderLoader);
	mRenderTargets = std::make_unique<RenderTargets>();
	
	mVertexShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "vs_main", nullptr,
		nvrhi::ShaderType::Vertex);
	mPixelShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "ps_main", nullptr,
		nvrhi::ShaderType::Pixel);

	nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
	bindlessLayoutDesc.visibility	  = nvrhi::ShaderType::All;
	bindlessLayoutDesc.firstSlot	  = 0;
	bindlessLayoutDesc.maxCapacity	  = 1024;
	bindlessLayoutDesc.registerSpaces = {
		nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
		nvrhi::BindingLayoutItem::Texture_SRV(2)};
	mBindlessLayout = getDevice()->createBindlessLayout(bindlessLayoutDesc);
	mDescriptorTableManager = std::make_shared<DescriptorTableManager>(
		getDevice(), mBindlessLayout);
	/* Initialize graphics scene data. */
	if (mScene->getLights().size() == 0) {
		Log(Warning, "The scene does not contain any light, adding a default sun light.");
		auto graph	  = mScene->getSceneGraph();
		auto sunLight = std::make_shared<DirectionalLight>(RGB(1), 2);
		graph->attachLeaf(graph->getRoot(), sunLight);
		sunLight->setDirection({0.5f, -0.8f, 0.5f});
		sunLight->setName("Sunlight");
	}
	// TODO: It seems possible to share the device buffer between graphics and CUDA/OptiX.
	mScene->initializeGraphicsScene(getDevice(), mDescriptorTableManager);
	std::shared_ptr<GraphicsScene> scene = mScene->mGraphicsScene;

	mCommandList = getDevice()->createCommandList();
	
	/* Create constant buffers */
	nvrhi::BufferDesc constantsBufferDesc;
	constantsBufferDesc.byteSize		 = sizeof(ViewConstants);
	constantsBufferDesc.debugName		 = "ViewConstants";
	constantsBufferDesc.isConstantBuffer = true;
	constantsBufferDesc.isVolatile		 = true;
	constantsBufferDesc.maxVersions		 = 16U;
	mViewConstants						 = getDevice()->createBuffer(constantsBufferDesc);

	constantsBufferDesc.byteSize  = sizeof(LightConstants);
	constantsBufferDesc.debugName = "LightData";
	mLightConstants				  = getDevice()->createBuffer(constantsBufferDesc);

	/* Create binding set */
	nvrhi::BindingSetDesc bindingSetDesc;
	bindingSetDesc.bindings = {
		nvrhi::BindingSetItem::ConstantBuffer(0, mViewConstants),
		nvrhi::BindingSetItem::ConstantBuffer(1, mLightConstants),
		nvrhi::BindingSetItem::PushConstants(2, sizeof(uint)),
		/* Mesh data constants (for indexing bindless buffers) */
		nvrhi::BindingSetItem::StructuredBuffer_SRV(0, scene->getGeometryBuffer()),
		/* Instance data constants (for transforming&indexing mesh) */
		nvrhi::BindingSetItem::StructuredBuffer_SRV(1, scene->getInstanceBuffer()),
		/* Material data constants (for indexing bindless buffers) */
		nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene->getMaterialBuffer()),
		/* Light data constants */
		nvrhi::BindingSetItem::StructuredBuffer_SRV(3, scene->getLightBuffer()),
		nvrhi::BindingSetItem::Sampler(0, mHelperPass->m_AnisotropicWrapSampler)
	};
	nvrhi::utils::CreateBindingSetAndLayout(
		getDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc,
		mBindingLayout, mBindingSet);
}

void BindlessRender::render(RenderContext *context) {
	PROFILE("Bindless Rendering");
	nvrhi::IFramebuffer *framebuffer = context->getFramebuffer();
	const auto &fbInfo				 = framebuffer->getFramebufferInfo();
	int sampleCount = 1;
	switch (mMSAA) {
		case MSAA::MSAA_2X: sampleCount = 2; break;
		case MSAA::MSAA_4X: sampleCount = 4; break;
		case MSAA::MSAA_8X: sampleCount = 8; break;
		default:;
	}
	if (!mRenderTargets ||
		mRenderTargets->isUpdateNeeded(Vector2i{fbInfo.width, fbInfo.height},
									   sampleCount)) {
		mRenderTargets->initialize(getDevice(),
			Vector2i{fbInfo.width, fbInfo.height}, sampleCount);
		mGraphicsPipeline = nullptr;
	}

	if (!mGraphicsPipeline) {
		/* Either first frame, or the backbuffer resized, or... */
		nvrhi::FramebufferDesc framebufferDesc;
		framebufferDesc.addColorAttachment(mRenderTargets->color, nvrhi::AllSubresources);
		framebufferDesc.setDepthAttachment(mRenderTargets->depth);
		mFramebuffer = getDevice()->createFramebuffer(framebufferDesc);

		nvrhi::GraphicsPipelineDesc pipelineDesc;
		pipelineDesc.VS				= mVertexShader;
		pipelineDesc.PS				= mPixelShader;
		pipelineDesc.primType		= nvrhi::PrimitiveType::TriangleList;
		pipelineDesc.bindingLayouts = {mBindingLayout, mBindlessLayout};
		pipelineDesc.renderState.rasterState.frontCounterClockwise = true;
		pipelineDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
		pipelineDesc.renderState.depthStencilState.depthTestEnable = true;
		pipelineDesc.renderState.depthStencilState.depthFunc =
			nvrhi::ComparisonFunc::LessOrEqual;
		mGraphicsPipeline = getDevice()->createGraphicsPipeline(pipelineDesc, mFramebuffer);
	}
	mCommandList->open();
	mRenderTargets->clear(mCommandList);

	/* Set view constants */
	ViewConstants viewConstants;
	Camera::SharedPtr camera	 = getScene()->getCamera();
	viewConstants.viewToClip	 = camera->getProjectionMatrix();
	viewConstants.worldToView	 = camera->getViewMatrix();
	viewConstants.worldToClip	 = camera->getViewProjectionMatrix();
	viewConstants.cameraPosition = camera->getPosition();
	mCommandList->writeBuffer(mViewConstants, &viewConstants, sizeof(viewConstants));

	/* Set light constsnts */
	LightConstants lightConstants;
	lightConstants.numLights	 = getScene()->getLights().size();
	lightConstants.ambientBottom = 0.2f;
	lightConstants.ambientTop	 = lightConstants.ambientBottom * RGB(0.3, 0.4, 0.3);
	mCommandList->writeBuffer(mLightConstants, &lightConstants, sizeof(lightConstants));

	/* Draw geometries. */
	nvrhi::GraphicsState state;
	state.pipeline	  = mGraphicsPipeline;
	state.framebuffer = mFramebuffer;
	state.bindings	  = {mBindingSet,
						 mDescriptorTableManager->GetDescriptorTable()};
	state.viewport.addViewportAndScissorRect(
		nvrhi::Viewport(0, fbInfo.width, 0, fbInfo.height, 0.f, 1.f));
	mCommandList->setGraphicsState(state);

	for (int instanceId = 0; instanceId < mScene->getMeshInstances().size(); instanceId++) {
		mCommandList->setPushConstants(&instanceId, sizeof(int));
		auto instance = mScene->getMeshInstances()[instanceId];
		auto mesh	  = instance->getMesh();

		nvrhi::DrawArguments args;
		args.instanceCount = 1;
		args.vertexCount   = mesh->indices.size() * 3;
		mCommandList->draw(args);
	}
	
	/* Blit framebuffer. */
	// We may not draw to the backbuffer directly due to unknown format and depth buffer.
	auto& resolvedColor = framebuffer->getDesc().colorAttachments[0].texture;
	if (sampleCount > 1)
		mCommandList->resolveTexture(resolvedColor, nvrhi::TextureSubresourceSet(0, 1, 0, 1),
			mRenderTargets->color, nvrhi::TextureSubresourceSet(0, 1, 0, 1));
	else mHelperPass->BlitTexture(mCommandList, framebuffer, mRenderTargets->color, mBindingCache.get());
	mCommandList->close();
	getDevice()->executeCommandList(mCommandList);
}

void BindlessRender::renderUI() {
	const char *msaa_mode[] = {"None", "MSAA_2X", "MSAA_4X", "MSAA_8X"};
	ui::Combo("MSAA", (int*) & mMSAA, msaa_mode, 4);
}

void BindlessRender::resize(const Vector2i &size) {
	mFramebuffer	  = nullptr;
	mGraphicsPipeline = nullptr;
	if(mBindingCache) mBindingCache->Clear();
}

KRR_REGISTER_PASS_DEF(BindlessRender);
NAMESPACE_END(krr)
