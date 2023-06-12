#include "bindless.h"
#include "window.h"
#include "vulkan/scene.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

class GBufferRenderTargets {
protected:
	Vector2i mSize{};
	uint mSampleCount{};

public:
	vkrhi::TextureHandle depth;
	vkrhi::TextureHandle diffuse;
	vkrhi::TextureHandle specular;
	vkrhi::TextureHandle normals;
	vkrhi::TextureHandle emissive;

	virtual bool isUpdateNeeded(Vector2i size, uint sampleCount) {
		return mSize != size || mSampleCount != sampleCount;
	}

	virtual void initialize(vkrhi::vulkan::IDevice* device,
		Vector2i size,
		uint sampleCount) {
		vkrhi::TextureDesc desc;
		desc.width			  = size[0];
		desc.height			  = size[1];
		desc.initialState	  = vkrhi::ResourceStates::DepthWrite;
		desc.keepInitialState = true;
		desc.isRenderTarget	  = true;
		desc.useClearValue	  = true;
		desc.isTypeless		  = true;
		desc.isUAV			  = false;
		desc.sampleCount	  = sampleCount;
		desc.mipLevels		  = 1;
		desc.format			  = vkrhi::Format::D24S8;
		desc.clearValue		  = vkrhi::Color(1.f);
		desc.debugName		  = "DepthBuffer";
		desc.dimension = sampleCount > 1 ? vkrhi::TextureDimension::Texture2DMS
										 : vkrhi::TextureDimension::Texture2D;
		depth				= device->createTexture(desc);
	
		mSize = size;
		mSampleCount = sampleCount;
	}

	virtual void clear(vkrhi::ICommandList* commandList) {
		const vkrhi::FormatInfo depthFormatInfo = vkrhi::getFormatInfo(depth->getDesc().format);
		commandList->clearDepthStencilTexture(depth, vkrhi::AllSubresources, 
			true, 1.f, depthFormatInfo.hasStencil, 0);
	}
};

class RenderTargets : public GBufferRenderTargets {
public:
	vkrhi::TextureHandle color;		// potentially a MSAA texture

	virtual void initialize(vkrhi::vulkan::IDevice *device, Vector2i size,
							uint sampleCount) override {
		GBufferRenderTargets::initialize(device, size, sampleCount);

		vkrhi::TextureDesc desc;
		desc.width			  = size[0];
		desc.height			  = size[1];
		desc.initialState	  = vkrhi::ResourceStates::RenderTarget;
		desc.keepInitialState = true;
		desc.isRenderTarget	  = true;
		desc.useClearValue	  = true;
		desc.sampleCount	  = sampleCount;
		desc.mipLevels		  = 1;
		desc.format			  = vkrhi::Format::RGBA32_FLOAT;
		desc.clearValue		  = vkrhi::Color(0.f);
		desc.debugName		  = "ColorBuffer";
		desc.dimension = sampleCount > 1 ? vkrhi::TextureDimension::Texture2DMS
										 : vkrhi::TextureDimension::Texture2D;
		color		   = device->createTexture(desc);
	}

	virtual void clear(vkrhi::ICommandList *commandList) {
		GBufferRenderTargets::clear(commandList);
		commandList->clearTextureFloat(color, vkrhi::AllSubresources,
											  vkrhi::Color(0.f));
	}
};

void BindlessRender::initialize() {
	mShaderLoader = std::make_shared<ShaderLoader>(getVulkanDevice());
	mBindingCache = std::make_shared<BindingCache>(getVulkanDevice());
	mHelperPass	  = std::make_shared<CommonRenderPasses>(getVulkanDevice(), mShaderLoader);
	mRenderTargets = std::make_unique<RenderTargets>();
	
	mVertexShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "vs_main", nullptr,
		vkrhi::ShaderType::Vertex);
	mPixelShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "ps_main", nullptr,
		vkrhi::ShaderType::Pixel);

	vkrhi::BindlessLayoutDesc bindlessLayoutDesc;
	bindlessLayoutDesc.visibility	  = vkrhi::ShaderType::All;
	bindlessLayoutDesc.firstSlot	  = 0;
	bindlessLayoutDesc.maxCapacity	  = 1024;
	bindlessLayoutDesc.registerSpaces = {
		nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
		nvrhi::BindingLayoutItem::Texture_SRV(2)};
	mBindlessLayout = getVulkanDevice()->createBindlessLayout(bindlessLayoutDesc);
	mDescriptorTableManager = std::make_shared<DescriptorTableManager>(
		getVulkanDevice(), mBindlessLayout);
	/* Initialize scene data on vulkan device. */
	// TODO: It seems possible to share the device buffer between vulkan and cuda/optix.
	mScene->initializeSceneVK(getVulkanDevice(), mDescriptorTableManager);
	std::shared_ptr<VKScene> scene = mScene->mSceneVK;

	mCommandList = getVulkanDevice()->createCommandList();
	
	/* Create view constant buffer */
	vkrhi::BufferDesc viewConstantsBufferDesc;
	viewConstantsBufferDesc.byteSize		 = sizeof(ViewConstants);
	viewConstantsBufferDesc.debugName		 = "ViewConstants";
	viewConstantsBufferDesc.isConstantBuffer = true;
	viewConstantsBufferDesc.isVolatile		 = true;
	viewConstantsBufferDesc.maxVersions		 = 16U;
	mViewConstants = getVulkanDevice()->createBuffer(viewConstantsBufferDesc);

	getVulkanDevice()->waitForIdle();
	/* Create binding set */
	vkrhi::BindingSetDesc bindingSetDesc;
	bindingSetDesc.bindings = {
		vkrhi::BindingSetItem::ConstantBuffer(0, mViewConstants),
		vkrhi::BindingSetItem::PushConstants(1, sizeof(uint)),
		/* Mesh data constants (for indexing bindless buffers) */
		vkrhi::BindingSetItem::StructuredBuffer_SRV(0, scene->getGeometryBuffer()),
		/* Instance data constants (for transforming&indexing mesh) */
		vkrhi::BindingSetItem::StructuredBuffer_SRV(1, scene->getInstanceBuffer()),
		/* Material data constants (for indexing bindless buffers) */
		vkrhi::BindingSetItem::StructuredBuffer_SRV(2, scene->getMaterialBuffer()),
		vkrhi::BindingSetItem::Sampler(0, mHelperPass->m_AnisotropicWrapSampler)
	};
	vkrhi::utils::CreateBindingSetAndLayout(
		getVulkanDevice(), vkrhi::ShaderType::All, 0, bindingSetDesc,
		mBindingLayout, mBindingSet);
}

void BindlessRender::render(RenderContext *context) {
	PROFILE("Bindless Rendering");
	vkrhi::IFramebuffer *framebuffer = context->getFramebuffer();
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
		mRenderTargets->initialize(getVulkanDevice(),
			Vector2i{fbInfo.width, fbInfo.height}, sampleCount);
		mGraphicsPipeline = nullptr;
	}

	if (!mGraphicsPipeline) {
		/* Either first frame, or the backbuffer resized, or... */
		vkrhi::FramebufferDesc framebufferDesc;
		framebufferDesc.addColorAttachment(mRenderTargets->color, vkrhi::AllSubresources);
		framebufferDesc.setDepthAttachment(mRenderTargets->depth);
		mFramebuffer = getVulkanDevice()->createFramebuffer(framebufferDesc);

		vkrhi::GraphicsPipelineDesc pipelineDesc;
		pipelineDesc.VS				= mVertexShader;
		pipelineDesc.PS				= mPixelShader;
		pipelineDesc.primType		= vkrhi::PrimitiveType::TriangleList;
		pipelineDesc.bindingLayouts = {mBindingLayout, mBindlessLayout};
		pipelineDesc.renderState.rasterState.frontCounterClockwise = true;
		pipelineDesc.renderState.rasterState.cullMode = vkrhi::RasterCullMode::None;
		pipelineDesc.renderState.depthStencilState.depthTestEnable = true;
		pipelineDesc.renderState.depthStencilState.depthFunc =
			vkrhi::ComparisonFunc::LessOrEqual;
		mGraphicsPipeline = getVulkanDevice()->createGraphicsPipeline(pipelineDesc, mFramebuffer);
	}
	mCommandList->open();
	mRenderTargets->clear(mCommandList);

	/* Set view constants */
	ViewConstants viewConstants;
	Camera::SharedPtr camera  = getScene()->getCamera();
	viewConstants.viewToClip  = camera->getProjectionMatrix();
	viewConstants.worldToView = camera->getViewMatrix();
	viewConstants.worldToClip = camera->getViewProjectionMatrix();

	mCommandList->writeBuffer(mViewConstants, &viewConstants,
							  sizeof(viewConstants));

	/* Draw geometries. */
	vkrhi::GraphicsState state;
	state.pipeline	  = mGraphicsPipeline;
	state.framebuffer = mFramebuffer;
	state.bindings	  = {mBindingSet,
						 mDescriptorTableManager->GetDescriptorTable()};
	state.viewport.addViewportAndScissorRect(
		vkrhi::Viewport(0, fbInfo.width, 0, fbInfo.height, 0.f, 1.f));
	mCommandList->setGraphicsState(state);

	for (int instanceId = 0; instanceId < mScene->getMeshInstances().size(); instanceId++) {
		mCommandList->setPushConstants(&instanceId, sizeof(int));
		auto instance = mScene->getMeshInstances()[instanceId];
		auto mesh	  = instance->getMesh();

		vkrhi::DrawArguments args;
		args.instanceCount = 1;
		args.vertexCount   = mesh->indices.size() * 3;
		mCommandList->draw(args);
	}
	
	/* Blit framebuffer. */
	// We may not draw to the backbuffer directly due to unknown format and depth buffer.
	auto& resolvedColor = framebuffer->getDesc().colorAttachments[0].texture;
	if (sampleCount > 1)
		mCommandList->resolveTexture(resolvedColor, vkrhi::TextureSubresourceSet(0, 1, 0, 1),
			mRenderTargets->color, vkrhi::TextureSubresourceSet(0, 1, 0, 1));
	else mHelperPass->BlitTexture(mCommandList, framebuffer, mRenderTargets->color, mBindingCache.get());
	mCommandList->close();
	getVulkanDevice()->executeCommandList(mCommandList);
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
KRR_NAMESPACE_END