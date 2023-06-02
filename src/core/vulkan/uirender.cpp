#include <stddef.h>

#include <window.h>
#include "render/profiler/profiler.h"
#include "uirender.h"
#include "shader.h"

KRR_NAMESPACE_BEGIN

bool UIRenderer::onMouseEvent(const io::MouseEvent &mouseEvent) {
	auto &io = ImGui::GetIO();
	switch (mouseEvent.type) {
		case io::MouseEvent::Type::Move:
			io.MousePos.x = float(mouseEvent.screenPos[0]);
			io.MousePos.y = float(mouseEvent.screenPos[1]);
			break;
		case io::MouseEvent::Type::Wheel:
			io.MouseWheel += float(mouseEvent.wheelDelta[1]);
			break;
		case io::MouseEvent::Type::LeftButtonDown:
			mouseDown[0] = io.MouseDown[0] = true;
			break;
		case io::MouseEvent::Type::LeftButtonUp:
			mouseDown[0] = false;
			break;
		case io::MouseEvent::Type::MiddleButtonDown:
			mouseDown[1] = io.MouseDown[1] = true;
			break;
		case io::MouseEvent::Type::MiddleButtonUp:
			mouseDown[1] = false;
			break;
		case io::MouseEvent::Type::RightButtonDown:
			mouseDown[2] = io.MouseDown[2] = true;
			break;
		case io::MouseEvent::Type::RightButtonUp:
			mouseDown[2] = false;
			break;
	}
	
	return io.WantCaptureMouse;
}

bool UIRenderer::onKeyEvent(const io::KeyboardEvent &keyEvent) {
	auto &io = ImGui::GetIO();

	if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed 
		|| keyEvent.type == io::KeyboardEvent::Type::KeyReleased) {
		bool keyIsDown{false};
		if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed)
			keyIsDown = true;
		int key = keyEvent.glfwKey;
		// update our internal state tracking for this key button
		keyDown[key] = keyIsDown;
		if (keyIsDown) io.KeysDown[key] = true;
		// if the key was pressed, update ImGui immediately
		// for key up events, ImGui state is only updated after the next frame
		// this ensures that short keypresses are not missed
	} else if (keyEvent.type == io::KeyboardEvent::Type::Input) {
		io.AddInputCharacter(keyEvent.codepoint);
	}
	return io.WantCaptureKeyboard;
}

bool UIRenderer::createFontTexture(nvrhi::ICommandList *commandList) {
	ImGuiIO &io = ImGui::GetIO();
	unsigned char *pixels;
	int width, height;

	io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

	{
		nvrhi::TextureDesc desc;
		desc.width	   = width;
		desc.height	   = height;
		desc.format	   = nvrhi::Format::RGBA8_UNORM;
		desc.debugName = "ImGui font texture";

		fontTexture = device->createTexture(desc);

		commandList->beginTrackingTextureState(
			fontTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::Common);

		if (fontTexture == nullptr) return false;

		commandList->writeTexture(fontTexture, 0, 0, pixels, width * 4);

		commandList->setPermanentTextureState(
			fontTexture, nvrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();

		io.Fonts->TexID = fontTexture;
	}

	{
		const auto desc =
			nvrhi::SamplerDesc()
				.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
				.setAllFilters(true);

		fontSampler = device->createSampler(desc);
		if (fontSampler == nullptr) return false;
	}

	return true;
}

void UIRenderer::initialize() {
	this->device = getVulkanDevice();
	if (!this->device) Log(Fatal, "Set device for UIRenderer before initialization!");
	
	auto shaderLoader = std::make_unique<ShaderLoader>(getVulkanDevice());

	m_commandList = device->createCommandList();
	m_commandList->open();

	vertexShader = shaderLoader->createShader(
		"common/shaders/imgui_vs.hlsl", "main", nullptr, nvrhi::ShaderType::Vertex);
	if (vertexShader == nullptr) {
		Log(Fatal, "error creating NVRHI vertex shader object\n");
	}

	pixelShader = shaderLoader->createShader(
		"common/shaders/imgui_ps.hlsl", "main", nullptr, nvrhi::ShaderType::Pixel);
	if (pixelShader == nullptr) {
		Log(Fatal, "error creating NVRHI pixel shader object\n");
	}

	// create attribute layout object
	nvrhi::VertexAttributeDesc vertexAttribLayout[] = {
		{"POSITION", nvrhi::Format::RG32_FLOAT, 1, 0, offsetof(ImDrawVert, pos),
		 sizeof(ImDrawVert), false},
		{"TEXCOORD", nvrhi::Format::RG32_FLOAT, 1, 0, offsetof(ImDrawVert, uv),
		 sizeof(ImDrawVert), false},
		{"COLOR", nvrhi::Format::RGBA8_UNORM, 1, 0, offsetof(ImDrawVert, col),
		 sizeof(ImDrawVert), false},
	};

	shaderAttribLayout = device->createInputLayout(
		vertexAttribLayout,
		sizeof(vertexAttribLayout) / sizeof(vertexAttribLayout[0]),
		vertexShader);

	if (!createFontTexture(m_commandList)) {
		Log(Fatal, "Failed to create font texture");
	}

	{
		nvrhi::BlendState blendState;
		blendState.targets[0]
			.setBlendEnable(true)
			.setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
			.setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
			.setSrcBlendAlpha(nvrhi::BlendFactor::InvSrcAlpha)
			.setDestBlendAlpha(nvrhi::BlendFactor::Zero);

		auto rasterState = nvrhi::RasterState()
							   .setFillSolid()
							   .setCullNone()
							   .setScissorEnable(true)
							   .setDepthClipEnable(true);

		auto depthStencilState =
			nvrhi::DepthStencilState()
				.disableDepthTest()
				.enableDepthWrite()
				.disableStencil()
				.setDepthFunc(nvrhi::ComparisonFunc::Always);

		nvrhi::RenderState renderState;
		renderState.blendState		  = blendState;
		renderState.depthStencilState = depthStencilState;
		renderState.rasterState		  = rasterState;

		nvrhi::BindingLayoutDesc layoutDesc;
		layoutDesc.visibility = nvrhi::ShaderType::All;
		layoutDesc.bindings	  = {
			  nvrhi::BindingLayoutItem::PushConstants(0, sizeof(float) * 2),
			  nvrhi::BindingLayoutItem::Texture_SRV(0),
			  nvrhi::BindingLayoutItem::Sampler(0)};
		bindingLayout = device->createBindingLayout(layoutDesc);

		basePSODesc.primType	   = nvrhi::PrimitiveType::TriangleList;
		basePSODesc.inputLayout	   = shaderAttribLayout;
		basePSODesc.VS			   = vertexShader;
		basePSODesc.PS			   = pixelShader;
		basePSODesc.renderState	   = renderState;
		basePSODesc.bindingLayouts = {bindingLayout};
	}
	
	auto &io = ImGui::GetIO();
	auto* font = io.Fonts->AddFontDefault();

	m_commandList->close();
	device->executeCommandList(m_commandList);
	device->waitForIdle();

	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard |
					  ImGuiConfigFlags_DockingEnable |
					  ImGuiConfigFlags_ViewportsEnable;

	auto &style			 = ui::GetStyle();
	style.WindowRounding = 5.f;
	ImGui::StyleColorsLight();

	/* Setup keyboard mapping for imgui */
	io.KeyMap[ImGuiKey_Tab]		   = GLFW_KEY_TAB;
	io.KeyMap[ImGuiKey_LeftArrow]  = GLFW_KEY_LEFT;
	io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
	io.KeyMap[ImGuiKey_UpArrow]	   = GLFW_KEY_UP;
	io.KeyMap[ImGuiKey_DownArrow]  = GLFW_KEY_DOWN;
	io.KeyMap[ImGuiKey_PageUp]	   = GLFW_KEY_PAGE_UP;
	io.KeyMap[ImGuiKey_PageDown]   = GLFW_KEY_PAGE_DOWN;
	io.KeyMap[ImGuiKey_Home]	   = GLFW_KEY_HOME;
	io.KeyMap[ImGuiKey_End]		   = GLFW_KEY_END;
	io.KeyMap[ImGuiKey_Delete]	   = GLFW_KEY_DELETE;
	io.KeyMap[ImGuiKey_Backspace]  = GLFW_KEY_BACKSPACE;
	io.KeyMap[ImGuiKey_Enter]	   = GLFW_KEY_ENTER;
	io.KeyMap[ImGuiKey_Escape]	   = GLFW_KEY_ESCAPE;
	io.KeyMap[ImGuiKey_A]		   = 'A';
	io.KeyMap[ImGuiKey_C]		   = 'C';
	io.KeyMap[ImGuiKey_V]		   = 'V';
	io.KeyMap[ImGuiKey_X]		   = 'X';
	io.KeyMap[ImGuiKey_Y]		   = 'Y';
	io.KeyMap[ImGuiKey_Z]		   = 'Z';
}

bool UIRenderer::reallocateBuffer(nvrhi::BufferHandle &buffer,
								   size_t requiredSize, size_t reallocateSize,
								   const bool indexBuffer) {
	if (buffer == nullptr ||
		size_t(buffer->getDesc().byteSize) < requiredSize) {
		nvrhi::BufferDesc desc;
		desc.byteSize	  = uint32_t(reallocateSize);
		desc.structStride = 0;
		desc.debugName =
			indexBuffer ? "ImGui index buffer" : "ImGui vertex buffer";
		desc.canHaveUAVs		= false;
		desc.isVertexBuffer		= !indexBuffer;
		desc.isIndexBuffer		= indexBuffer;
		desc.isDrawIndirectArgs = false;
		desc.isVolatile			= false;
		desc.initialState	  = indexBuffer ? nvrhi::ResourceStates::IndexBuffer
											: nvrhi::ResourceStates::VertexBuffer;
		desc.keepInitialState = true;

		buffer = device->createBuffer(desc);

		if (!buffer) {
			return false;
		}
	}

	return true;
}

void UIRenderer::tick(float elapsedTimeSeconds) {
	ImGuiIO &io		   = ImGui::GetIO();
	io.DeltaTime	   = elapsedTimeSeconds;
	io.MouseDrawCursor = false;
}

void UIRenderer::beginFrame() {
	int width, height;
	float scaleX, scaleY;

	getDeviceManager()->GetWindowDimensions(width, height);
	getDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);

	ImGuiIO &io					 = ImGui::GetIO();
	io.DisplaySize				 = ImVec2(float(width), float(height));
	io.DisplayFramebufferScale.x = scaleX;
	io.DisplayFramebufferScale.y = scaleY;

	io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
	io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
	io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
	io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
	ImGui::NewFrame();
}

void UIRenderer::endFrame() {
	// reconcile input key states
	auto &io = ImGui::GetIO();
	for (size_t i = 0; i < mouseDown.size(); i++) 
		if (io.MouseDown[i] == true && mouseDown[i] == false) 
			io.MouseDown[i] = false;
	for (size_t i = 0; i < keyDown.size(); i++) 
		if (io.KeysDown[i] == true && keyDown[i] == false) 
			io.KeysDown[i] = false;
}

nvrhi::IGraphicsPipeline *UIRenderer::getPSO(nvrhi::IFramebuffer *fb) {
	if (pso) return pso;
	pso = device->createGraphicsPipeline(basePSODesc, fb);
	assert(pso);
	return pso;
}

nvrhi::IBindingSet *UIRenderer::getBindingSet(nvrhi::ITexture *texture) {
	auto iter = bindingsCache.find(texture);
	if (iter != bindingsCache.end()) {
		return iter->second;
	}

	nvrhi::BindingSetDesc desc;

	desc.bindings = {nvrhi::BindingSetItem::PushConstants(0, sizeof(float) * 2),
					 nvrhi::BindingSetItem::Texture_SRV(0, texture),
					 nvrhi::BindingSetItem::Sampler(0, fontSampler)};

	nvrhi::BindingSetHandle binding;
	binding = device->createBindingSet(desc, bindingLayout);
	assert(binding);

	bindingsCache[texture] = binding;
	return binding;
}

bool UIRenderer::updateGeometry(nvrhi::ICommandList *commandList) {
	ImDrawData *drawData = ImGui::GetDrawData();

	// create/resize vertex and index buffers if needed
	if (!reallocateBuffer(
			vertexBuffer, drawData->TotalVtxCount * sizeof(ImDrawVert),
			(drawData->TotalVtxCount + 5000) * sizeof(ImDrawVert), false)) {
		return false;
	}

	if (!reallocateBuffer(
			indexBuffer, drawData->TotalIdxCount * sizeof(ImDrawIdx),
			(drawData->TotalIdxCount + 5000) * sizeof(ImDrawIdx), true)) {
		return false;
	}

	vtxBuffer.resize(vertexBuffer->getDesc().byteSize / sizeof(ImDrawVert));
	idxBuffer.resize(indexBuffer->getDesc().byteSize / sizeof(ImDrawIdx));

	// copy and convert all vertices into a single contiguous buffer
	ImDrawVert *vtxDst = &vtxBuffer[0];
	ImDrawIdx *idxDst  = &idxBuffer[0];

	for (int n = 0; n < drawData->CmdListsCount; n++) {
		const ImDrawList *cmdList = drawData->CmdLists[n];

		memcpy(vtxDst, cmdList->VtxBuffer.Data,
			   cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(idxDst, cmdList->IdxBuffer.Data,
			   cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));

		vtxDst += cmdList->VtxBuffer.Size;
		idxDst += cmdList->IdxBuffer.Size;
	}

	commandList->writeBuffer(vertexBuffer, &vtxBuffer[0],
							 vertexBuffer->getDesc().byteSize);
	commandList->writeBuffer(indexBuffer, &idxBuffer[0],
							 indexBuffer->getDesc().byteSize);

	return true;
}

void UIRenderer::render(RenderFrame::SharedPtr frame) {
	PROFILE("UI Render");
	ImGui::Render();

	ImDrawData *drawData = ImGui::GetDrawData();
	const auto &io		 = ImGui::GetIO();

	m_commandList->open();
	m_commandList->beginMarker("ImGUI");

	if (!updateGeometry(m_commandList)) {
		Log(Error, "UIRender::Failed to update geometry for imgui render.");
		return;
	}
		
	// handle DPI scaling
	drawData->ScaleClipRects(io.DisplayFramebufferScale);

	float invDisplaySize[2] = {1.f / io.DisplaySize.x, 1.f / io.DisplaySize.y};

	// set up graphics state
	nvrhi::GraphicsState drawState;

	drawState.framebuffer = frame->getFramebuffer();
	assert(drawState.framebuffer);

	drawState.pipeline = getPSO(drawState.framebuffer);

	drawState.viewport.viewports.push_back(
		nvrhi::Viewport(io.DisplaySize.x * io.DisplayFramebufferScale.x,
						io.DisplaySize.y * io.DisplayFramebufferScale.y));
	drawState.viewport.scissorRects.resize(1); // updated below

	nvrhi::VertexBufferBinding vbufBinding;
	vbufBinding.buffer = vertexBuffer;
	vbufBinding.slot   = 0;
	vbufBinding.offset = 0;
	drawState.vertexBuffers.push_back(vbufBinding);

	drawState.indexBuffer.buffer = indexBuffer;
	drawState.indexBuffer.format =
		(sizeof(ImDrawIdx) == 2 ? nvrhi::Format::R16_UINT
								: nvrhi::Format::R32_UINT);
	drawState.indexBuffer.offset = 0;

	// render command lists
	int vtxOffset = 0;
	int idxOffset = 0;
	for (int n = 0; n < drawData->CmdListsCount; n++) {
		const ImDrawList *cmdList = drawData->CmdLists[n];
		for (int i = 0; i < cmdList->CmdBuffer.Size; i++) {
			const ImDrawCmd *pCmd = &cmdList->CmdBuffer[i];

			if (pCmd->UserCallback) {
				pCmd->UserCallback(cmdList, pCmd);
			} else {
				drawState.bindings = {
					getBindingSet((nvrhi::ITexture *) pCmd->TextureId)};
				assert(drawState.bindings[0]);

				drawState.viewport.scissorRects[0] =
					nvrhi::Rect(int(pCmd->ClipRect.x), int(pCmd->ClipRect.z),
								int(pCmd->ClipRect.y), int(pCmd->ClipRect.w));

				nvrhi::DrawArguments drawArguments;
				drawArguments.vertexCount		  = pCmd->ElemCount;
				drawArguments.startIndexLocation  = idxOffset;
				drawArguments.startVertexLocation = vtxOffset;

				m_commandList->setGraphicsState(drawState);
				m_commandList->setPushConstants(invDisplaySize,
												sizeof(invDisplaySize));
				m_commandList->drawIndexed(drawArguments);
			}

			idxOffset += pCmd->ElemCount;
		}

		vtxOffset += cmdList->VtxBuffer.Size;
	}

	m_commandList->endMarker();
	m_commandList->close();
	device->executeCommandList(m_commandList);
}

void UIRenderer::resizing() { pso = nullptr; }


KRR_NAMESPACE_END