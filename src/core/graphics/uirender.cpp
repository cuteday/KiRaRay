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

namespace {

ImGuiKey getImGuiKey(int key) {
	if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) return ImGuiKey(ImGuiKey_0 + key - GLFW_KEY_0);
	if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) return ImGuiKey(ImGuiKey_A + key - GLFW_KEY_A);
	if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F24) return ImGuiKey(ImGuiKey_F1 + key - GLFW_KEY_F1);
	if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_9) return ImGuiKey(ImGuiKey_Keypad0 + key - GLFW_KEY_KP_0);
	switch (key) {
	case GLFW_KEY_TAB: return ImGuiKey_Tab;
	case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
	case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
	case GLFW_KEY_UP: return ImGuiKey_UpArrow;
	case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
	case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
	case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
	case GLFW_KEY_HOME: return ImGuiKey_Home;
	case GLFW_KEY_END: return ImGuiKey_End;
	case GLFW_KEY_INSERT: return ImGuiKey_Insert;
	case GLFW_KEY_DELETE: return ImGuiKey_Delete;
	case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
	case GLFW_KEY_SPACE: return ImGuiKey_Space;
	case GLFW_KEY_ENTER: return ImGuiKey_Enter;
	case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
	case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
	case GLFW_KEY_COMMA: return ImGuiKey_Comma;
	case GLFW_KEY_MINUS: return ImGuiKey_Minus;
	case GLFW_KEY_PERIOD: return ImGuiKey_Period;
	case GLFW_KEY_SLASH: return ImGuiKey_Slash;
	case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
	case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
	case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
	case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
	case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
	case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
	case GLFW_KEY_CAPS_LOCK: return ImGuiKey_CapsLock;
	case GLFW_KEY_SCROLL_LOCK: return ImGuiKey_ScrollLock;
	case GLFW_KEY_NUM_LOCK: return ImGuiKey_NumLock;
	case GLFW_KEY_PRINT_SCREEN: return ImGuiKey_PrintScreen;
	case GLFW_KEY_PAUSE: return ImGuiKey_Pause;
	case GLFW_KEY_KP_DECIMAL: return ImGuiKey_KeypadDecimal;
	case GLFW_KEY_KP_DIVIDE: return ImGuiKey_KeypadDivide;
	case GLFW_KEY_KP_MULTIPLY: return ImGuiKey_KeypadMultiply;
	case GLFW_KEY_KP_SUBTRACT: return ImGuiKey_KeypadSubtract;
	case GLFW_KEY_KP_ADD: return ImGuiKey_KeypadAdd;
	case GLFW_KEY_KP_ENTER: return ImGuiKey_KeypadEnter;
	case GLFW_KEY_KP_EQUAL: return ImGuiKey_KeypadEqual;
	case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
	case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
	case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
	case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
	case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
	case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
	case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
	case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
	case GLFW_KEY_MENU: return ImGuiKey_Menu;
	case GLFW_KEY_WORLD_1:
	case GLFW_KEY_WORLD_2: return ImGuiKey_Oem102;
	default: return ImGuiKey_None;
	}
}

void updateModifiers(ImGuiIO &input, const io::InputModifiers &mods) {
	input.AddKeyEvent(ImGuiMod_Ctrl, mods.isCtrlDown);
	input.AddKeyEvent(ImGuiMod_Shift, mods.isShiftDown);
	input.AddKeyEvent(ImGuiMod_Alt, mods.isAltDown);
	input.AddKeyEvent(ImGuiMod_Super, mods.isSuperDown);
}

}

UIRenderer::~UIRenderer() {
	if (!mContext) return;
	ImGuiContext *previous = ImGui::GetCurrentContext();
	ImGui::SetCurrentContext(mContext);
	for (ImTextureData *texture : ImGui::GetPlatformIO().Textures)
		if (texture->RefCount == 1) destroyTexture(texture);
	ImGui::GetIO().BackendRendererUserData = nullptr;
	ImGui::GetIO().BackendRendererName = nullptr;
	ImGui::GetIO().BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
	ImGui::GetPlatformIO().ClearRendererHandlers();
	ImGui::DestroyContext(mContext);
	if (previous != mContext) ImGui::SetCurrentContext(previous);
}

bool UIRenderer::onMouseEvent(const io::MouseEvent &mouseEvent) {
	if (!mContext) return false;
	ImGui::SetCurrentContext(mContext);
	auto &input = ImGui::GetIO();
	using Type = io::MouseEvent::Type;
	int button = -1;
	bool down = false;
	switch (mouseEvent.type) {
	case Type::Move:
		input.AddMousePosEvent(mouseEvent.screenPos[0], mouseEvent.screenPos[1]);
		break;
	case Type::Wheel:
		input.AddMouseWheelEvent(mouseEvent.wheelDelta[0], mouseEvent.wheelDelta[1]);
		break;
	case Type::LeftButtonDown:
	case Type::LeftButtonUp:
		button = 0;
		down = mouseEvent.type == Type::LeftButtonDown;
		break;
	case Type::RightButtonDown:
	case Type::RightButtonUp:
		button = 1;
		down = mouseEvent.type == Type::RightButtonDown;
		break;
	case Type::MiddleButtonDown:
	case Type::MiddleButtonUp:
		button = 2;
		down = mouseEvent.type == Type::MiddleButtonDown;
		break;
	}
	if (button >= 0) {
		updateModifiers(input, mouseEvent.mods);
		input.AddMouseButtonEvent(button, down);
	}
	return input.WantCaptureMouse;
}

bool UIRenderer::onKeyEvent(const io::KeyboardEvent &keyEvent) {
	if (!mContext) return false;
	ImGui::SetCurrentContext(mContext);
	auto &input = ImGui::GetIO();
	if (keyEvent.type == io::KeyboardEvent::Type::Input) input.AddInputCharacter(keyEvent.codepoint);
	else {
		updateModifiers(input, keyEvent.mods);
		ImGuiKey key = getImGuiKey(keyEvent.glfwKey);
		if (key != ImGuiKey_None) input.AddKeyEvent(key, keyEvent.type == io::KeyboardEvent::Type::KeyPressed);
	}
	return input.WantCaptureKeyboard;
}

void UIRenderer::onWindowFocus(int focused) {
	if (!mContext) return;
	ImGui::SetCurrentContext(mContext);
	ImGui::GetIO().AddFocusEvent(focused != 0);
}

void UIRenderer::destroyTexture(ImTextureData *texture) {
	auto it = mTextures.find(texture);
	if (it != mTextures.end()) {
		bindingsCache.erase(it->second);
		mTextures.erase(it);
	}
	texture->BackendUserData = nullptr;
	texture->SetTexID(ImTextureID_Invalid);
	texture->SetStatus(ImTextureStatus_Destroyed);
}

void UIRenderer::updateTexture(nvrhi::ICommandList *commandList, ImTextureData *texture) {
	if (texture->Status == ImTextureStatus_WantCreate) {
		if (texture->Format != ImTextureFormat_RGBA32)
			throw std::runtime_error("UI textures require RGBA32 pixels.");
		nvrhi::TextureDesc desc;
		desc.width = texture->Width;
		desc.height = texture->Height;
		desc.format = nvrhi::Format::RGBA8_UNORM;
		desc.debugName = "ImGui texture";
		desc.initialState = nvrhi::ResourceStates::ShaderResource;
		desc.keepInitialState = true;
		nvrhi::TextureHandle image = device->createTexture(desc);
		if (!image) throw std::runtime_error("Failed to create UI texture.");
		commandList->writeTexture(image, 0, 0, texture->GetPixels(), texture->GetPitch());
		mTextures.emplace(texture, image);
		texture->BackendUserData = image.Get();
		texture->SetTexID(ImTextureID(reinterpret_cast<uintptr_t>(image.Get())));
		texture->SetStatus(ImTextureStatus_OK);
	} else if (texture->Status == ImTextureStatus_WantUpdates) {
		auto image = mTextures.at(texture);
		const ImTextureRect &rect = texture->UpdateRect;
		if (rect.w && rect.h) {
			nvrhi::TextureDesc desc;
			desc.width = rect.w;
			desc.height = rect.h;
			desc.format = nvrhi::Format::RGBA8_UNORM;
			desc.debugName = "ImGui texture upload";
			auto staging = device->createStagingTexture(desc, nvrhi::CpuAccessMode::Write);
			if (!staging) throw std::runtime_error("Failed to create UI texture upload.");
			size_t rowPitch = 0;
			auto *pixels = static_cast<uint8_t *>(device->mapStagingTexture(
				staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Write, &rowPitch));
			if (!pixels) throw std::runtime_error("Failed to map UI texture upload.");
			const auto *source = static_cast<const uint8_t *>(texture->GetPixelsAt(rect.x, rect.y));
			for (int row = 0; row < rect.h; ++row)
				memcpy(pixels + row * rowPitch, source + row * texture->GetPitch(), size_t(rect.w) * 4);
			device->unmapStagingTexture(staging);
			commandList->copyTexture(image,
				nvrhi::TextureSlice().setOrigin(rect.x, rect.y).setSize(rect.w, rect.h, 1),
				staging, nvrhi::TextureSlice());
		}
		texture->SetStatus(ImTextureStatus_OK);
	} else if (texture->Status == ImTextureStatus_WantDestroy && texture->UnusedFrames > 0) {
		// Submitted command lists retain resources until their GPU work completes.
		destroyTexture(texture);
	}
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
	io.BackendRendererUserData = this;
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
	ImGui::StyleColorsLight();
	ImGui::GetStyle().WindowRounding = 5.f;
	ImGui::GetStyle().FontSizeBase = 15.f;
	mBaseStyle = ImGui::GetStyle();
	ImFontConfig fontConfig;
	fontConfig.OversampleH = 2;
	fontConfig.OversampleV = 1;
	const fs::path fontPath = fs::path(KRR_PROJECT_DIR) / "common/assets/fonts/Roboto-Medium.ttf";
	if (fs::is_regular_file(fontPath))
		io.FontDefault = io.Fonts->AddFontFromFileTTF(fontPath.u8string().c_str(), 15.f, &fontConfig);
	else {
		Log(Warning, "UI font is missing; using the embedded fallback: %s", fontPath.string().c_str());
		io.FontDefault = io.Fonts->AddFontDefaultVector(&fontConfig);
	}
	auto &platformIO = ImGui::GetPlatformIO();
	platformIO.Renderer_TextureMaxWidth = platformIO.Renderer_TextureMaxHeight = 4096;
	platformIO.DrawCallback_ResetRenderState = [](const ImDrawList *, const ImDrawCmd *) {
		auto *renderer = static_cast<UIRenderer *>(ImGui::GetIO().BackendRendererUserData);
		renderer->mPointSampler = false;
		renderer->m_commandList->clearState();
	};
	platformIO.DrawCallback_SetSamplerLinear = [](const ImDrawList *, const ImDrawCmd *) {
		static_cast<UIRenderer *>(ImGui::GetIO().BackendRendererUserData)->mPointSampler = false;
	};
	platformIO.DrawCallback_SetSamplerNearest = [](const ImDrawList *, const ImDrawCmd *) {
		static_cast<UIRenderer *>(ImGui::GetIO().BackendRendererUserData)->mPointSampler = true;
	};
	for (size_t i = 0; i < samplers.size(); ++i) {
		samplers[i] = device->createSampler(nvrhi::SamplerDesc()
			.setAllAddressModes(nvrhi::SamplerAddressMode::Clamp).setAllFilters(i == 0));
		if (!samplers[i]) throw std::runtime_error("Failed to create UI sampler.");
	}
	
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
	auto &io = ImGui::GetIO();
	io.DeltaTime = mPreviousTime >= 0.f && elapsedTimeSeconds > mPreviousTime
		? elapsedTimeSeconds - mPreviousTime : 1.f / 60.f;
	mPreviousTime = elapsedTimeSeconds;
	io.MouseDrawCursor = false;
}

void UIRenderer::beginFrame(RenderContext* context) {
	ImGui::SetCurrentContext(mContext);
	auto &io = ImGui::GetIO();
	int width, height, framebufferWidth, framebufferHeight;
	glfwGetWindowSize(getDeviceManager()->getWindow(), &width, &height);
	getDeviceManager()->getFrameSize(framebufferWidth, framebufferHeight);
	io.DisplaySize = ImVec2(float(width), float(height));
	io.DisplayFramebufferScale = ImVec2(width > 0 ? float(framebufferWidth) / width : 1.f,
		height > 0 ? float(framebufferHeight) / height : 1.f);
	float scaleX, scaleY;
	getDeviceManager()->getDPIScaleInfo(scaleX, scaleY);
	float framebufferScale = io.DisplayFramebufferScale.y;
	if (!std::isfinite(scaleY) || scaleY <= 0.f) scaleY = 1.f;
	if (!std::isfinite(framebufferScale) || framebufferScale <= 0.f) framebufferScale = 1.f;
	float dpiScale = scaleY / framebufferScale;
	if (std::abs(dpiScale - mDpiScale) > .001f) {
		ImGui::GetStyle() = mBaseStyle;
		ImGui::GetStyle().ScaleAllSizes(dpiScale);
		ImGui::GetStyle().FontScaleDpi = dpiScale;
		mDpiScale = dpiScale;
	}
	ImGui::NewFrame();
}

nvrhi::IGraphicsPipeline *UIRenderer::getPSO(nvrhi::IFramebuffer *fb) {
	if (pso) return pso;
	pso = device->createGraphicsPipeline(basePSODesc, fb);
	assert(pso);
	return pso;
}

nvrhi::IBindingSet *UIRenderer::getBindingSet(nvrhi::ITexture *texture) {
	auto &binding = bindingsCache[texture][mPointSampler ? 1 : 0];
	if (binding) return binding;

	nvrhi::BindingSetDesc desc;

	desc.bindings = {nvrhi::BindingSetItem::PushConstants(0, sizeof(float) * 4),
					 nvrhi::BindingSetItem::Texture_SRV(0, texture),
					 nvrhi::BindingSetItem::Sampler(0, samplers[mPointSampler ? 1 : 0])};
	binding = device->createBindingSet(desc, bindingLayout);
	assert(binding);

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
	if (!drawData) return;
	int framebufferWidth = int(drawData->DisplaySize.x * drawData->FramebufferScale.x);
	int framebufferHeight = int(drawData->DisplaySize.y * drawData->FramebufferScale.y);
	bool hasGeometry = drawData->DisplaySize.x > 0.f && drawData->DisplaySize.y > 0.f &&
		framebufferWidth > 0 && framebufferHeight > 0 &&
		drawData->TotalVtxCount > 0 && drawData->TotalIdxCount > 0;
	bool hasUploads = false;
	if (drawData->Textures) {
		for (ImTextureData *texture : *drawData->Textures) {
			if (texture->Status == ImTextureStatus_WantDestroy && texture->UnusedFrames > 0)
				destroyTexture(texture);
			hasUploads |= texture->Status == ImTextureStatus_WantCreate || texture->Status == ImTextureStatus_WantUpdates;
		}
	}
	if (!hasGeometry && !hasUploads) return;

	m_commandList->open();
	m_commandList->beginMarker("ImGUI");
	mPointSampler = false;
	auto submit = [&] {
		m_commandList->endMarker();
		m_commandList->close();
		device->executeCommandList(m_commandList);
	};
	if (drawData->Textures)
		for (ImTextureData *texture : *drawData->Textures)
			if (texture->Status != ImTextureStatus_OK) updateTexture(m_commandList, texture);
	if (!hasGeometry) {
		submit();
		return;
	}

	if (!updateGeometry(m_commandList)) {
		submit();
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
				pCmd->UserCallback(cmdList, pCmd);
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
					getBindingSet(reinterpret_cast<nvrhi::ITexture *>(uintptr_t(pCmd->GetTexID())))};
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

	submit();
}

void UIRenderer::resizing() { pso = nullptr; }


NAMESPACE_END(krr)
