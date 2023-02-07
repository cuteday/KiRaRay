#include "helperpass.h"
#include "binding.h"
#include "shader.h"

KRR_NAMESPACE_BEGIN

struct BlitConstants {
	Vector2f sourceOrigin;
	Vector2f sourceSize;

	Vector2f targetOrigin;
	Vector2f targetSize;

	float sharpenFactor;
};

CommonRenderPasses::CommonRenderPasses(nvrhi::IDevice *device,
									   std::shared_ptr<ShaderLoader> ShaderLoader) :
	m_Device(device) {
	{
		std::vector<ShaderMacro> VsMacros;
		VsMacros.push_back(ShaderMacro("QUAD_Z", "0"));
		m_FullscreenVS = ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/fullscreen_vs.hlsl", "main", &VsMacros,
													 nvrhi::ShaderType::Vertex);

		VsMacros[0].definition = "1";
		m_FullscreenAtOneVS = ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/fullscreen_vs.hlsl", "main", &VsMacros,
														  nvrhi::ShaderType::Vertex);
	}

	m_RectVS =
		ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/rect_vs.hlsl", "main", nullptr, nvrhi::ShaderType::Vertex);

	std::vector<ShaderMacro> blitMacros = {ShaderMacro("TEXTURE_ARRAY", "0")};
	m_BlitPS =
		ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/blit_ps.hlsl", "main", &blitMacros, nvrhi::ShaderType::Pixel);
	m_SharpenPS				 = ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/sharpen_ps.hlsl", "main", &blitMacros,
														   nvrhi::ShaderType::Pixel);
	blitMacros[0].definition = "1"; // TEXTURE_ARRAY
	m_BlitArrayPS =
		ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/blit_ps.hlsl", "main", &blitMacros, nvrhi::ShaderType::Pixel);
	m_SharpenArrayPS = ShaderLoader->createShader("src/misc/samples/simple-rhi/shaders/sharpen_ps.hlsl", "main", &blitMacros,
												   nvrhi::ShaderType::Pixel);

	auto samplerDesc = nvrhi::SamplerDesc().setAllFilters(false).setAllAddressModes(
		nvrhi::SamplerAddressMode::Clamp);
	m_PointClampSampler = m_Device->createSampler(samplerDesc);

	samplerDesc.setAllFilters(true);
	m_LinearClampSampler = m_Device->createSampler(samplerDesc);

	samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
	m_LinearWrapSampler = m_Device->createSampler(samplerDesc);

	samplerDesc.setMaxAnisotropy(16);
	m_AnisotropicWrapSampler = m_Device->createSampler(samplerDesc);

	{
		unsigned int blackImage = 0xff000000;
		unsigned int grayImage	= 0xff808080;
		unsigned int whiteImage = 0xffffffff;

		nvrhi::TextureDesc textureDesc;
		textureDesc.format	  = nvrhi::Format::RGBA8_UNORM;
		textureDesc.width	  = 1;
		textureDesc.height	  = 1;
		textureDesc.mipLevels = 1;

		textureDesc.debugName = "BlackTexture";
		m_BlackTexture		  = m_Device->createTexture(textureDesc);

		textureDesc.debugName = "GrayTexture";
		m_GrayTexture		  = m_Device->createTexture(textureDesc);

		textureDesc.debugName = "WhiteTexture";
		m_WhiteTexture		  = m_Device->createTexture(textureDesc);

		textureDesc.dimension = nvrhi::TextureDimension::TextureCubeArray;
		textureDesc.debugName = "BlackCubeMapArray";
		textureDesc.arraySize = 6;
		m_BlackCubeMapArray	  = m_Device->createTexture(textureDesc);

		textureDesc.dimension = nvrhi::TextureDimension::Texture2DArray;
		textureDesc.debugName = "BlackTexture2DArray";
		textureDesc.arraySize = 6;
		m_BlackTexture2DArray = m_Device->createTexture(textureDesc);
		textureDesc.debugName = "WhiteTexture2DArray";
		m_WhiteTexture2DArray = m_Device->createTexture(textureDesc);

		// Write the textures using a temporary CL

		nvrhi::CommandListHandle commandList = m_Device->createCommandList();
		commandList->open();

		commandList->beginTrackingTextureState(m_BlackTexture, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);
		commandList->beginTrackingTextureState(m_GrayTexture, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);
		commandList->beginTrackingTextureState(m_WhiteTexture, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);
		commandList->beginTrackingTextureState(m_BlackCubeMapArray, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);
		commandList->beginTrackingTextureState(m_BlackTexture2DArray, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);
		commandList->beginTrackingTextureState(m_WhiteTexture2DArray, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);

		commandList->writeTexture(m_BlackTexture, 0, 0, &blackImage, 0);
		commandList->writeTexture(m_GrayTexture, 0, 0, &grayImage, 0);
		commandList->writeTexture(m_WhiteTexture, 0, 0, &whiteImage, 0);

		for (uint32_t arraySlice = 0; arraySlice < 6; arraySlice += 1) {
			commandList->writeTexture(m_BlackTexture2DArray, arraySlice, 0, &blackImage, 0);
			commandList->writeTexture(m_WhiteTexture2DArray, arraySlice, 0, &whiteImage, 0);
			commandList->writeTexture(m_BlackCubeMapArray, arraySlice, 0, &blackImage, 0);
		}

		commandList->setPermanentTextureState(m_BlackTexture,
											  nvrhi::ResourceStates::ShaderResource);
		commandList->setPermanentTextureState(m_GrayTexture, nvrhi::ResourceStates::ShaderResource);
		commandList->setPermanentTextureState(m_WhiteTexture,
											  nvrhi::ResourceStates::ShaderResource);
		commandList->setPermanentTextureState(m_BlackCubeMapArray,
											  nvrhi::ResourceStates::ShaderResource);
		commandList->setPermanentTextureState(m_BlackTexture2DArray,
											  nvrhi::ResourceStates::ShaderResource);
		commandList->setPermanentTextureState(m_WhiteTexture2DArray,
											  nvrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();

		commandList->close();
		m_Device->executeCommandList(commandList);
	}

	{
		nvrhi::BindingLayoutDesc layoutDesc;
		layoutDesc.visibility = nvrhi::ShaderType::All;
		layoutDesc.bindings	  = {nvrhi::BindingLayoutItem::PushConstants(0, sizeof(BlitConstants)),
								 nvrhi::BindingLayoutItem::Texture_SRV(0),
								 nvrhi::BindingLayoutItem::Sampler(0)};

		m_BlitBindingLayout = m_Device->createBindingLayout(layoutDesc);
	}
}

static bool IsSupportedBlitDimension(nvrhi::TextureDimension dimension) {
	return dimension == nvrhi::TextureDimension::Texture2D ||
		   dimension == nvrhi::TextureDimension::Texture2DArray ||
		   dimension == nvrhi::TextureDimension::TextureCube ||
		   dimension == nvrhi::TextureDimension::TextureCubeArray;
}

static bool IsTextureArray(nvrhi::TextureDimension dimension) {
	return dimension == nvrhi::TextureDimension::Texture2DArray ||
		   dimension == nvrhi::TextureDimension::TextureCube ||
		   dimension == nvrhi::TextureDimension::TextureCubeArray;
}

void CommonRenderPasses::BlitTexture(nvrhi::ICommandList *commandList, const BlitParameters &params,
									 BindingCache *bindingCache) {
	assert(commandList);
	assert(params.targetFramebuffer);
	assert(params.sourceTexture);

	const nvrhi::FramebufferDesc &targetFramebufferDesc = params.targetFramebuffer->getDesc();
	assert(targetFramebufferDesc.colorAttachments.size() == 1);
	assert(targetFramebufferDesc.colorAttachments[0].valid());
	assert(!targetFramebufferDesc.depthAttachment.valid());

	const nvrhi::FramebufferInfoEx &fbinfo = params.targetFramebuffer->getFramebufferInfo();
	const nvrhi::TextureDesc &sourceDesc   = params.sourceTexture->getDesc();

	assert(IsSupportedBlitDimension(sourceDesc.dimension));
	bool isTextureArray = IsTextureArray(sourceDesc.dimension);

	nvrhi::Viewport targetViewport = params.targetViewport;
	if (targetViewport.width() == 0 && targetViewport.height() == 0) {
		// If no viewport is specified, create one based on the framebuffer dimensions.
		// Note that the FB dimensions may not be the same as target texture dimensions, in case a
		// non-zero mip level is used.
		targetViewport = nvrhi::Viewport(float(fbinfo.width), float(fbinfo.height));
	}

	nvrhi::IShader *shader = nullptr;
	switch (params.sampler) {
		case BlitSampler::Point:
		case BlitSampler::Linear:
			shader = isTextureArray ? m_BlitArrayPS : m_BlitPS;
			break;
		case BlitSampler::Sharpen:
			shader = isTextureArray ? m_SharpenArrayPS : m_SharpenPS;
			break;
		default:
			assert(false);
	}

	nvrhi::GraphicsPipelineHandle &pso =
		m_BlitPsoCache[PsoCacheKey{fbinfo, shader, params.blendState}];
	if (!pso) {
		nvrhi::GraphicsPipelineDesc psoDesc;
		psoDesc.bindingLayouts = {m_BlitBindingLayout};
		psoDesc.VS			   = m_RectVS;
		psoDesc.PS			   = shader;
		psoDesc.primType	   = nvrhi::PrimitiveType::TriangleStrip;
		psoDesc.renderState.rasterState.setCullNone();
		psoDesc.renderState.depthStencilState.depthTestEnable = false;
		psoDesc.renderState.depthStencilState.stencilEnable	  = false;
		psoDesc.renderState.blendState.targets[0]			  = params.blendState;

		pso = m_Device->createGraphicsPipeline(psoDesc, params.targetFramebuffer);
	}

	nvrhi::BindingSetDesc bindingSetDesc;
	{
		auto sourceDimension = sourceDesc.dimension;
		if (sourceDimension == nvrhi::TextureDimension::TextureCube ||
			sourceDimension == nvrhi::TextureDimension::TextureCubeArray)
			sourceDimension = nvrhi::TextureDimension::Texture2DArray;

		auto sourceSubresources =
			nvrhi::TextureSubresourceSet(params.sourceMip, 1, params.sourceArraySlice, 1);

		bindingSetDesc.bindings = {
			nvrhi::BindingSetItem::PushConstants(0, sizeof(BlitConstants)),
			nvrhi::BindingSetItem::Texture_SRV(0, params.sourceTexture)
				.setSubresources(sourceSubresources)
				.setDimension(sourceDimension),
			nvrhi::BindingSetItem::Sampler(0, params.sampler == BlitSampler::Point
												  ? m_PointClampSampler
												  : m_LinearClampSampler)};
	}

	// If a binding cache is provided, get the binding set from the cache.
	// Otherwise, create one and then release it.
	nvrhi::BindingSetHandle sourceBindingSet;
	if (bindingCache)
		sourceBindingSet = bindingCache->GetOrCreateBindingSet(bindingSetDesc, m_BlitBindingLayout);
	else
		sourceBindingSet = m_Device->createBindingSet(bindingSetDesc, m_BlitBindingLayout);

	nvrhi::GraphicsState state;
	state.pipeline	  = pso;
	state.framebuffer = params.targetFramebuffer;
	state.bindings	  = {sourceBindingSet};
	state.viewport.addViewport(targetViewport);
	state.viewport.addScissorRect(nvrhi::Rect(targetViewport));
	state.blendConstantColor = params.blendConstantColor;

	BlitConstants blitConstants = {};
	blitConstants.sourceOrigin	= Vector2f(params.sourceBox.min());
	blitConstants.sourceSize	= params.sourceBox.diagonal();
	blitConstants.targetOrigin	= Vector2f(params.targetBox.min());
	blitConstants.targetSize	= params.targetBox.diagonal();

	commandList->setGraphicsState(state);

	commandList->setPushConstants(&blitConstants, sizeof(blitConstants));

	nvrhi::DrawArguments args;
	args.instanceCount = 1;
	args.vertexCount   = 4;
	commandList->draw(args);
}

void CommonRenderPasses::BlitTexture(nvrhi::ICommandList *commandList,
									 nvrhi::IFramebuffer *targetFramebuffer,
									 nvrhi::ITexture *sourceTexture, BindingCache *bindingCache) {
	assert(commandList);
	assert(targetFramebuffer);
	assert(sourceTexture);

	BlitParameters params;
	params.targetFramebuffer = targetFramebuffer;
	params.sourceTexture	 = sourceTexture;
	BlitTexture(commandList, params, bindingCache);
}

KRR_NAMESPACE_END