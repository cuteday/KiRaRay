#include "bindless.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

void BindlessRender::initialize() {
	mShaderLoader = std::make_shared<ShaderLoader>(getVulkanDevice());
	mBindingCache = std::make_shared<BindingCache>(getVulkanDevice());
	mHelperPass	  = std::make_shared<CommonRenderPasses>(getVulkanDevice(), mShaderLoader);
	
	mVertexShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "vs_main", nullptr,
		vkrhi::ShaderType::Vertex);
	mPixelShader = mShaderLoader->createShader(
		"src/render/rasterize/shaders/bindless.hlsl", "ps_main", nullptr,
		vkrhi::ShaderType::Pixel);

	vkrhi::BindlessLayoutDesc bindlessLayoutDesc;
	bindlessLayoutDesc.visibility = vkrhi::ShaderType::All;
	bindlessLayoutDesc.firstSlot   = 0;
	bindlessLayoutDesc.maxCapacity = 1024;
	bindlessLayoutDesc.registerSpaces = {
		nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
		nvrhi::BindingLayoutItem::Texture_SRV(2)
	};
	mBindlessLayout = getVulkanDevice()->createBindlessLayout(bindlessLayoutDesc);
	
	mDescriptorTableManager = std::make_shared<DescriptorTableManager>(
		getVulkanDevice(), mBindlessLayout);

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

	};
	vkrhi::utils::CreateBindingSetAndLayout(
		getVulkanDevice(), vkrhi::ShaderType::All, 0, bindingSetDesc,
		mBindingLayout, mBindingSet);
}

void BindlessRender::render(RenderFrame::SharedPtr frame) {
	vkrhi::IFramebuffer *framebuffer = frame->getFramebuffer();
	const auto &fbInfo				 = framebuffer->getFramebufferInfo();

	if (!mGraphicsPipeline) {
		/* Either first frame, or the backbuffer resized, or... */
		vkrhi::TextureDesc textureDesc;
		textureDesc.width			 = fbInfo.width;
		textureDesc.height			 = fbInfo.height;
		textureDesc.dimension		 = vkrhi::TextureDimension::Texture2D;
		textureDesc.keepInitialState = true;
		textureDesc.isRenderTarget	 = true;

		textureDesc.debugName = "ColorBuffer";
		textureDesc.format	  = vkrhi::Format::RGBA32_FLOAT;
		textureDesc.initialState = vkrhi::ResourceStates::RenderTarget;
		mColorBuffer		  = getVulkanDevice()->createTexture(textureDesc);
		
		textureDesc.debugName	 = "DepthBuffer";
		textureDesc.format = vkrhi::Format::D24S8;
		textureDesc.initialState = vkrhi::ResourceStates::DepthWrite;
		mDepthBuffer = getVulkanDevice()->createTexture(textureDesc);
		
		vkrhi::FramebufferDesc framebufferDesc;
		framebufferDesc.addColorAttachment(mColorBuffer, vkrhi::AllSubresources);
		framebufferDesc.setDepthAttachment(mDepthBuffer);
		mFramebuffer = getVulkanDevice()->createFramebuffer(framebufferDesc);

		vkrhi::GraphicsPipelineDesc pipelineDesc;
		pipelineDesc.VS = mVertexShader;
		pipelineDesc.PS = mPixelShader;
		pipelineDesc.primType = vkrhi::PrimitiveType::TriangleList;
		pipelineDesc.bindingLayouts = {mBindingLayout, mBindlessLayout};
		pipelineDesc.renderState.rasterState.frontCounterClockwise = true;
		pipelineDesc.renderState.rasterState.cullMode = vkrhi::RasterCullMode::Back;
		pipelineDesc.renderState.depthStencilState.depthTestEnable = true;
		pipelineDesc.renderState.depthStencilState.depthFunc =
			vkrhi::ComparisonFunc::GreaterOrEqual;
		mGraphicsPipeline = getVulkanDevice()->createGraphicsPipeline(pipelineDesc, mFramebuffer);
	}
	
	mCommandList->open();
	mCommandList->clearTextureFloat(mColorBuffer, vkrhi::AllSubresources,
									vkrhi::Color(0));
	mCommandList->clearDepthStencilTexture(mDepthBuffer, vkrhi::AllSubresources,
										   true, 0, true, 0);

	/* Set view constants */
	ViewConstants viewConstants;
	mCommandList->writeBuffer(mViewConstants, &viewConstants,
							  sizeof(viewConstants));

	/* Draw geometries. */
	vkrhi::GraphicsState state;
	state.pipeline = mGraphicsPipeline;
	state.framebuffer = mFramebuffer;
	state.bindings	  = {mBindingSet,
						 mDescriptorTableManager->GetDescriptorTable()};
	mCommandList->setGraphicsState(state);
	
	/* Blit framebuffer. */
	// We may not draw to the backbuffer directly due to unknown format and depth buffer.
	mHelperPass->BlitTexture(mCommandList, framebuffer, mColorBuffer, mBindingCache.get());
	mCommandList->close();
	getVulkanDevice()->executeCommandList(mCommandList);
}

void BindlessRender::renderUI() { 
	ui::Text("Hello from bindless render pass");
}

void BindlessRender::resize(const Vector2i &size) {
	mDepthBuffer = nullptr;
	mColorBuffer = nullptr;
	mFramebuffer = nullptr;
	mGraphicsPipeline = nullptr;
	mBindingCache->Clear();
}

KRR_REGISTER_PASS_DEF(BindlessRender);
KRR_NAMESPACE_END