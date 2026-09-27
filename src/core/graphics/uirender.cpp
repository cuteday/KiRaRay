#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <file.h>

#include <graphics/device.h>
#include "render/profiler/profiler.h"
#include "uirender.h"
#include "shader.h"

NAMESPACE_BEGIN(krr)

UIRenderer::~UIRenderer() {
	if (mContext) ImGui::DestroyContext(mContext);
}

bool UIRenderer::onMouseEvent(const io::MouseEvent &mouseEvent) {
	if (!mContext) return false;
	ImGui::SetCurrentContext(mContext);
	auto &io = ImGui::GetIO();
	switch (mouseEvent.type) {
		case io::MouseEvent::Type::Move:
			io.AddMousePosEvent(float(mouseEvent.screenPos[0]), float(mouseEvent.screenPos[1]));
			break;
		case io::MouseEvent::Type::Wheel:
			io.AddMouseWheelEvent(float(mouseEvent.wheelDelta[0]), float(mouseEvent.wheelDelta[1]));
			break;
		case io::MouseEvent::Type::LeftButtonDown:
			io.AddMouseButtonEvent(0, true);
			break;
		case io::MouseEvent::Type::LeftButtonUp:
			io.AddMouseButtonEvent(0, false);
			break;
		case io::MouseEvent::Type::MiddleButtonDown:
			io.AddMouseButtonEvent(2, true);
			break;
		case io::MouseEvent::Type::MiddleButtonUp:
			io.AddMouseButtonEvent(2, false);
			break;
		case io::MouseEvent::Type::RightButtonDown:
			io.AddMouseButtonEvent(1, true);
			break;
		case io::MouseEvent::Type::RightButtonUp:
			io.AddMouseButtonEvent(1, false);
			break;
	}
	
	return io.WantCaptureMouse;
}

bool UIRenderer::onKeyEvent(const io::KeyboardEvent &keyEvent) {
	if (!mContext) return false;
	ImGui::SetCurrentContext(mContext);
	auto &io = ImGui::GetIO();

	if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed 
		|| keyEvent.type == io::KeyboardEvent::Type::KeyReleased) {
		bool keyIsDown{false};
		if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed)
			keyIsDown = true;
		int key = keyEvent.glfwKey;
		if (key < 0 || key >= int(keyDown.size())) return io.WantCaptureKeyboard;
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

void UIRenderer::onWindowFocus(int focused) {
	if (!mContext) return;
	ImGui::SetCurrentContext(mContext);
	ImGui::GetIO().AddFocusEvent(focused != 0);
	if (!focused) keyDown.fill(false);
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
		if (fontTexture == nullptr) return false;

		commandList->beginTrackingTextureState(
			fontTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::Common);

		commandList->writeTexture(fontTexture, 0, 0, pixels, width * 4);

		commandList->setPermanentTextureState(
			fontTexture, nvrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();

		io.Fonts->TexID = fontTexture;
	}

	if (!fontSampler) {
		const auto desc =
			nvrhi::SamplerDesc()
				.setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
				.setAllFilters(true);

		fontSampler = device->createSampler(desc);
		if (fontSampler == nullptr) return false;
	}

	return true;
}

void UIRenderer::updateFont(nvrhi::ICommandList *commandList, float fontScale, float framebufferScale) {
	auto &io = ImGui::GetIO();
	bindingsCache.clear();
	fontTexture = nullptr;
	io.FontDefault = nullptr;
	io.Fonts->Clear();
	ImFontConfig config;
	config.OversampleH = 2;
	config.OversampleV = 1;
	const fs::path fontPath = fs::path(KRR_PROJECT_DIR) / "common/assets/fonts/Roboto-Medium.ttf";
	if (fs::is_regular_file(fontPath)) {
		io.FontDefault = io.Fonts->AddFontFromFileTTF(fontPath.u8string().c_str(), 15.f * fontScale, &config);
	} else {
		Log(Warning, "UI font is missing; using the embedded fallback: %s", fontPath.string().c_str());
		config.SizePixels = 13.f * fontScale;
		config.OversampleH = 1;
		config.PixelSnapH = true;
		io.FontDefault = io.Fonts->AddFontDefault(&config);
	}
	io.FontGlobalScale = 1.f / framebufferScale;
	ImGui::GetStyle() = mBaseStyle;
	ImGui::GetStyle().ScaleAllSizes(fontScale / framebufferScale);
	if (!createFontTexture(commandList)) Log(Fatal, "Failed to create font texture");
	mFontScale = fontScale;
	mFramebufferScale = framebufferScale;
}

void UIRenderer::initialize() {
	if (mContext) throw std::logic_error("UIRenderer is already initialized.");
	this->device = getDevice();
	if (!this->device) Log(Fatal, "Set device for UIRenderer before initialization!");
	IMGUI_CHECKVERSION();
	mContext = ImGui::CreateContext();
	ImGui::SetCurrentContext(mContext);
	auto &io = ImGui::GetIO();
	io.BackendRendererName = "kiraray_nvrhi";
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
	ImGui::StyleColorsLight();
	ImGui::GetStyle().WindowRounding = 5.f;
	mBaseStyle = ImGui::GetStyle();
	
	auto shaderLoader = std::make_unique<ShaderLoader>(getDevice());

	m_commandList = device->createCommandList();

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

	{
		nvrhi::BlendState blendState;
		blendState.targets[0]
			.setBlendEnable(true)
			.setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
			.setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
			.setSrcBlendAlpha(nvrhi::BlendFactor::One)
			.setDestBlendAlpha(nvrhi::BlendFactor::InvSrcAlpha);

		auto rasterState = nvrhi::RasterState()
							   .setFillSolid()
							   .setCullNone()
							   .setScissorEnable(true)
							   .setDepthClipEnable(true);

		auto depthStencilState =
			nvrhi::DepthStencilState()
				.disableDepthTest()
				.disableDepthWrite()
				.disableStencil()
				.setDepthFunc(nvrhi::ComparisonFunc::Always);

		nvrhi::RenderState renderState;
		renderState.blendState		  = blendState;
		renderState.depthStencilState = depthStencilState;
		renderState.rasterState		  = rasterState;

		nvrhi::BindingLayoutDesc layoutDesc;
		layoutDesc.visibility = nvrhi::ShaderType::All;
		layoutDesc.bindings	  = {
			  nvrhi::BindingLayoutItem::PushConstants(0, sizeof(float) * 4),
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
	ImGui::SetCurrentContext(mContext);
	ImGuiIO &io		   = ImGui::GetIO();
	io.DeltaTime = mPreviousTime >= 0.f && elapsedTimeSeconds > mPreviousTime
		? elapsedTimeSeconds - mPreviousTime : 1.f / 60.f;
	mPreviousTime = elapsedTimeSeconds;
	io.MouseDrawCursor = false;
}

void UIRenderer::beginFrame(RenderContext* context) {
	ImGui::SetCurrentContext(mContext);
	int width, height, framebufferWidth, framebufferHeight;
	float scaleX, scaleY;

	glfwGetWindowSize(getDeviceManager()->getWindow(), &width, &height);
	getDeviceManager()->getFrameSize(framebufferWidth, framebufferHeight);
	getDeviceManager()->getDPIScaleInfo(scaleX, scaleY);

	ImGuiIO &io					 = ImGui::GetIO();
	io.DisplaySize				 = ImVec2(float(width), float(height));
	io.DisplayFramebufferScale = ImVec2(width > 0 ? float(framebufferWidth) / width : 1.f,
		height > 0 ? float(framebufferHeight) / height : 1.f);
	float framebufferScale = io.DisplayFramebufferScale.y;
	if (!std::isfinite(scaleY) || scaleY <= 0.f) scaleY = 1.f;
	if (!std::isfinite(framebufferScale) || framebufferScale <= 0.f) framebufferScale = 1.f;
	if (std::abs(scaleY - mFontScale) > .001f ||
		std::abs(framebufferScale - mFramebufferScale) > .001f) {
		if (fontTexture) device->waitForIdle();
		m_commandList->open();
		updateFont(m_commandList, scaleY, framebufferScale);
		m_commandList->close();
		device->executeCommandList(m_commandList);
	}

	io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
	io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
	io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
	io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
	ImGui::NewFrame();
}

void UIRenderer::endFrame(RenderContext* context) {
	ImGui::SetCurrentContext(mContext);
	// reconcile input key states
	auto &io = ImGui::GetIO();
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

	desc.bindings = {nvrhi::BindingSetItem::PushConstants(0, sizeof(float) * 4),
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

	vtxBuffer.resize(drawData->TotalVtxCount);
	idxBuffer.resize(drawData->TotalIdxCount);

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
							 vtxBuffer.size() * sizeof(ImDrawVert));
	commandList->writeBuffer(indexBuffer, &idxBuffer[0],
							 idxBuffer.size() * sizeof(ImDrawIdx));

	return true;
}

void UIRenderer::render(RenderContext *context) {
	PROFILE("UI Render");
	ImGui::SetCurrentContext(mContext);
	ImGui::Render();

	ImDrawData *drawData = ImGui::GetDrawData();
	if (!drawData || drawData->DisplaySize.x <= 0.f || drawData->DisplaySize.y <= 0.f ||
		drawData->TotalVtxCount == 0 || drawData->TotalIdxCount == 0) return;
	int framebufferWidth = int(drawData->DisplaySize.x * drawData->FramebufferScale.x);
	int framebufferHeight = int(drawData->DisplaySize.y * drawData->FramebufferScale.y);
	if (framebufferWidth <= 0 || framebufferHeight <= 0) return;

	m_commandList->open();
	m_commandList->beginMarker("ImGUI");

	if (!updateGeometry(m_commandList)) {
		m_commandList->endMarker();
		m_commandList->close();
		Log(Error, "UIRender::Failed to update geometry for imgui render.");
		return;
	}

	float constants[4] = {1.f / drawData->DisplaySize.x, 1.f / drawData->DisplaySize.y,
		drawData->DisplayPos.x, drawData->DisplayPos.y};

	// set up graphics state
	nvrhi::GraphicsState drawState;

	drawState.framebuffer = context->getFramebuffer();
	assert(drawState.framebuffer);

	drawState.pipeline = getPSO(drawState.framebuffer);
	const auto &framebufferInfo = drawState.framebuffer->getFramebufferInfo();
	int clipWidth = std::min(framebufferWidth, int(framebufferInfo.width));
	int clipHeight = std::min(framebufferHeight, int(framebufferInfo.height));

	drawState.viewport.viewports.push_back(
		nvrhi::Viewport(float(framebufferWidth), float(framebufferHeight)));
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
				if (pCmd->UserCallback == ImDrawCallback_ResetRenderState)
					m_commandList->clearState();
				else pCmd->UserCallback(cmdList, pCmd);
			} else {
				ImVec2 clipMin((pCmd->ClipRect.x - drawData->DisplayPos.x) * drawData->FramebufferScale.x,
					(pCmd->ClipRect.y - drawData->DisplayPos.y) * drawData->FramebufferScale.y);
				ImVec2 clipMax((pCmd->ClipRect.z - drawData->DisplayPos.x) * drawData->FramebufferScale.x,
					(pCmd->ClipRect.w - drawData->DisplayPos.y) * drawData->FramebufferScale.y);
				clipMin.x = std::max(clipMin.x, 0.f);
				clipMin.y = std::max(clipMin.y, 0.f);
				clipMax.x = std::min(clipMax.x, float(clipWidth));
				clipMax.y = std::min(clipMax.y, float(clipHeight));
				if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y || pCmd->ElemCount == 0) continue;
				nvrhi::Rect scissor(int(clipMin.x), int(clipMax.x), int(clipMin.y), int(clipMax.y));
				if (scissor.maxX <= scissor.minX || scissor.maxY <= scissor.minY) continue;
				drawState.bindings = {
					getBindingSet((nvrhi::ITexture *) pCmd->TextureId)};
				assert(drawState.bindings[0]);

				drawState.viewport.scissorRects[0] = scissor;

				nvrhi::DrawArguments drawArguments;
				drawArguments.vertexCount		  = pCmd->ElemCount;
				drawArguments.startIndexLocation  = idxOffset + pCmd->IdxOffset;
				drawArguments.startVertexLocation = vtxOffset + pCmd->VtxOffset;

				m_commandList->setGraphicsState(drawState);
				m_commandList->setPushConstants(constants, sizeof(constants));
				m_commandList->drawIndexed(drawArguments);
			}
		}

		vtxOffset += cmdList->VtxBuffer.Size;
		idxOffset += cmdList->IdxBuffer.Size;
	}

	m_commandList->endMarker();
	m_commandList->close();
	device->executeCommandList(m_commandList);
}

void UIRenderer::resizing() { pso = nullptr; }


NAMESPACE_END(krr)
