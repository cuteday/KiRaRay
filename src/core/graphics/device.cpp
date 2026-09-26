#include <cstdio>
#include <exception>
#include <thread>

#include "graphics/device.h"
#include "graphics/device_backend.h"
#include "graphics/interop.h"
#include "graphics/binding.h"
#include "graphics/helperpass.h"
#include "imgui.h"
#include <nvrhi/validation.h>
#include "logger.h"
#include "device/context.h"

#include "render/profiler/profiler.h"

#ifdef _WINDOWS
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")
#endif


NAMESPACE_BEGIN(krr)

namespace {
using namespace krr::io;

class ApiCallbacks {
public:
	static void windowIconifyCallback(GLFWwindow *window, int iconified) {
		DeviceManager *manager =
			reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
		manager->onWindowIconify(iconified);
	}

	static void windowFocusCallback(GLFWwindow *window, int focused) {
		DeviceManager *manager =
			reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
		manager->onWindowFocus(focused);
	}

	static void windowRefreshCallback(GLFWwindow *window) {
		DeviceManager *manager =
			reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
		manager->onWindowRefresh();
	}

	static void windowCloseCallback(GLFWwindow *window) {
		DeviceManager *manager =
			reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
		manager->onWindowClose();

	}

	static void windowPosCallback(GLFWwindow *window, int xpos, int ypos) {
		DeviceManager *manager =
			reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
		manager->onWindowPosUpdate(xpos, ypos);
	}

	static void keyboardCallback(GLFWwindow *pGlfwWindow, int key, int scanCode, int action,
								 int modifiers) {
		KeyboardEvent event;
		if (prepareKeyboardEvent(key, action, modifiers, event)) {
			DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
			if (manager != nullptr) {
				manager->onKeyEvent(event);
			}
		}
	}

	static void charInputModsCallback(GLFWwindow *pGlfwWindow, uint32_t input, int modifiers) {
		KeyboardEvent event;
		event.type		= KeyboardEvent::Type::Input;
		event.codepoint = input;
		event.mods		= getInputModifiers(modifiers);

		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
		if (manager != nullptr) {
			manager->onKeyEvent(event);
		}
	}

	static void mouseMoveCallback(GLFWwindow *pGlfwWindow, double mouseX, double mouseY) {
		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
		if (manager != nullptr) {
			// Prepare the mouse data
			MouseEvent event;
			event.type		 = MouseEvent::Type::Move;
			event.pos		 = calcMousePos(mouseX, mouseY, manager->getMouseScale());
			event.screenPos	 = Vector2f(mouseX, mouseY);
			event.wheelDelta = Vector2f(0, 0);

			manager->onMouseEvent(event);
		}
	}

	static void mouseButtonCallback(GLFWwindow *pGlfwWindow, int button, int action,
									int modifiers) {
		MouseEvent event;
		// Prepare the mouse data
		switch (button) {
			case GLFW_MOUSE_BUTTON_LEFT:
				event.type = (action == GLFW_PRESS || action == GLFW_REPEAT) ?
					MouseEvent::Type::LeftButtonDown : MouseEvent::Type::LeftButtonUp;
				break;
			case GLFW_MOUSE_BUTTON_MIDDLE:
				event.type = (action == GLFW_PRESS || action == GLFW_REPEAT) ?
					MouseEvent::Type::MiddleButtonDown : MouseEvent::Type::MiddleButtonUp;
				break;
			case GLFW_MOUSE_BUTTON_RIGHT:
				event.type = (action == GLFW_PRESS || action == GLFW_REPEAT) ?
					MouseEvent::Type::RightButtonDown : MouseEvent::Type::RightButtonUp;
				break;
			default:
				// Other keys are not supported
				break;
		}

		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
		if (manager != nullptr) {
			// Modifiers
			event.mods = getInputModifiers(modifiers);
			double x, y;
			glfwGetCursorPos(pGlfwWindow, &x, &y);
			event.pos = calcMousePos(x, y, manager->getMouseScale());

			manager->onMouseEvent(event);
		}
	}

	static void mouseWheelCallback(GLFWwindow *pGlfwWindow, double scrollX, double scrollY) {
		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
		if (manager != nullptr) {
			MouseEvent event;
			event.type = MouseEvent::Type::Wheel;
			double x, y;
			glfwGetCursorPos(pGlfwWindow, &x, &y);
			event.pos		 = calcMousePos(x, y, manager->getMouseScale());
			event.wheelDelta = Vector2f(float(scrollX), float(scrollY));

			manager->onMouseEvent(event);
		}
	}

	static void errorCallback(int errorCode, const char *pDescription) {
		Log(Error, "GLFW Error[%d]: %s", errorCode, pDescription);
	}

private:
	static inline InputModifiers getInputModifiers(int mask) {
		InputModifiers mods;
		mods.isAltDown	 = (mask & GLFW_MOD_ALT) != 0;
		mods.isCtrlDown	 = (mask & GLFW_MOD_CONTROL) != 0;
		mods.isShiftDown = (mask & GLFW_MOD_SHIFT) != 0;
		return mods;
	}

	// calculates the mouse pos in sreen [0, 1]^2
	static inline Vector2f calcMousePos(double xPos, double yPos, const Vector2f &mouseScale) {
		Vector2f pos = Vector2f(float(xPos), float(yPos));
		pos			 = pos.cwiseProduct(mouseScale);
		return pos;
	}

	static inline bool prepareKeyboardEvent(int key, int action, int modifiers,
											KeyboardEvent &event) {
		if (action == GLFW_REPEAT || key == GLFW_KEY_UNKNOWN) return false;

		event.type = (action == GLFW_RELEASE ? KeyboardEvent::Type::KeyReleased
											 : KeyboardEvent::Type::KeyPressed);
		event.glfwKey = key;
		event.key	  = glfwToKey(key);
		event.mods	  = getInputModifiers(modifiers);
		return true;
	}
};

struct DefaultMessageCallback : public nvrhi::IMessageCallback {
	static DefaultMessageCallback &GetInstance() {
		static DefaultMessageCallback Instance;
		return Instance;
	}

	void message(nvrhi::MessageSeverity severity, const char *messageText) override {
		Log::Level krrSeverity = Log::Level::Info;
		switch (severity) {
			case nvrhi::MessageSeverity::Info:
				krrSeverity = Log::Level::Info;
				break;
			case nvrhi::MessageSeverity::Warning:
				krrSeverity = Log::Level::Warning;
				break;
			case nvrhi::MessageSeverity::Error:
				krrSeverity = Log::Level::Error;
				break;
			case nvrhi::MessageSeverity::Fatal:
				krrSeverity = Log::Level::Fatal;
				break;
		}
		logMessage(krrSeverity, messageText);
	}
};
}

DeviceManager::DeviceManager() = default;
DeviceManager::~DeviceManager() {
	try { shutdown(); }
	catch (const std::exception &error) { std::fprintf(stderr, "Device cleanup: %s\n", error.what()); }
	catch (...) { std::fprintf(stderr, "Device cleanup failed.\n"); }
}

bool DeviceManager::createWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
												   const char *windowTitle) {
#ifdef _WINDOWS
	if (params.enablePerMonitorDPI) {
		// this needs to happen before glfwInit in order to override GLFW behavior
		SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
	} else {
		SetProcessDpiAwareness(PROCESS_DPI_UNAWARE);
	}
#endif

	if (!glfwInit()) {
		return false;
	}
	mGlfwInitialized = true;
	mHeadless = false;

	this->mDeviceParams = params;
	mRequestedVSync	 = params.vsyncEnabled;

	glfwSetErrorCallback(ApiCallbacks::errorCallback);

	glfwDefaultWindowHints();

	glfwWindowHint(GLFW_REFRESH_RATE, params.refreshRate);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Ignored for fullscreen

	mWindow = glfwCreateWindow(
		params.backBufferWidth, params.backBufferHeight, windowTitle ? windowTitle : "",
		params.startFullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

	if (mWindow == nullptr) {
		return false;
	}

	if (params.startFullscreen) {
		glfwSetWindowMonitor(mWindow, glfwGetPrimaryMonitor(), 0, 0,
							 mDeviceParams.backBufferWidth, mDeviceParams.backBufferHeight,
							 mDeviceParams.refreshRate);
	} else {
		int fbWidth = 0, fbHeight = 0;
		glfwGetFramebufferSize(mWindow, &fbWidth, &fbHeight);
		mDeviceParams.backBufferWidth	= fbWidth;
		mDeviceParams.backBufferHeight = fbHeight;
	}

	if (windowTitle) mWindowTitle = windowTitle;

	glfwSetWindowUserPointer(mWindow, this);

	if (params.windowPosX != -1 && params.windowPosY != -1) {
		glfwSetWindowPos(mWindow, params.windowPosX, params.windowPosY);
	}

	if (params.startMaximized) {
		glfwMaximizeWindow(mWindow);
	}

	glfwSetWindowPosCallback(mWindow, ApiCallbacks::windowPosCallback);
	glfwSetWindowCloseCallback(mWindow, ApiCallbacks::windowCloseCallback);
	glfwSetWindowRefreshCallback(mWindow, ApiCallbacks::windowRefreshCallback);
	glfwSetWindowFocusCallback(mWindow, ApiCallbacks::windowFocusCallback);
	glfwSetWindowIconifyCallback(mWindow, ApiCallbacks::windowIconifyCallback);
	glfwSetKeyCallback(mWindow, ApiCallbacks::keyboardCallback);
	glfwSetCursorPosCallback(mWindow, ApiCallbacks::mouseMoveCallback);
	glfwSetMouseButtonCallback(mWindow, ApiCallbacks::mouseButtonCallback);
	glfwSetScrollCallback(mWindow, ApiCallbacks::mouseWheelCallback);
	glfwSetCharModsCallback(mWindow, ApiCallbacks::charInputModsCallback);

	if (!createDeviceAndSwapChain()) return false;

	glfwShowWindow(mWindow);

	// reset the back buffer size state to enforce a resize event
	mDeviceParams.backBufferWidth	= 0;
	mDeviceParams.backBufferHeight = 0;

	updateWindowSize();
	mNvrhiDevice->waitForIdle();

	auto ctx = ImGui::CreateContext();
	ImGui::SetCurrentContext(ctx);

	return true;
}

bool DeviceManager::createHeadlessDevice(const DeviceCreationParameters &params) {
	mDeviceParams = params;
	mHeadless = true;
	try {
		if (!createDeviceAndSwapChain()) {
			shutdown();
			return false;
		}
		mRenderContext->resize(getFrameSize());
		return true;
	} catch (...) {
		shutdown();
		throw;
	}
}

void DeviceManager::addRenderPassToFront(RenderPass::SharedPtr pRenderPass) {
	mRenderPasses.remove(pRenderPass);
	mRenderPasses.push_front(pRenderPass);
	pRenderPass->setDeviceManager(this);
}

void DeviceManager::addRenderPassToBack(RenderPass::SharedPtr pRenderPass) {
	mRenderPasses.remove(pRenderPass);
	mRenderPasses.push_back(pRenderPass);
	pRenderPass->setDeviceManager(this);
}

void DeviceManager::removeRenderPass(RenderPass::SharedPtr pRenderPass) {
	mRenderPasses.remove(pRenderPass);
}

void DeviceManager::backBufferResizing() {
	mSwapChainFramebuffers.clear();
	if (mBindingCache) mBindingCache->Clear();
	for (auto it : mRenderPasses) {
		it->resizing();
	}
}

void DeviceManager::backBufferResized() {
	for (auto it : mRenderPasses) {
		it->resize({int(mDeviceParams.backBufferWidth),
					int(mDeviceParams.backBufferHeight)});
	}

	uint32_t backBufferCount = getBackBufferCount();
	mSwapChainFramebuffers.resize(backBufferCount);
	for (uint32_t index = 0; index < backBufferCount; index++) {
		mSwapChainFramebuffers[index] = getDevice()->createFramebuffer(
			nvrhi::FramebufferDesc().addColorAttachment(getBackBuffer(index)));
	}
	// resize render targets
	mRenderContext->resize(
		{int(mDeviceParams.backBufferWidth), int(mDeviceParams.backBufferHeight)});
}

void DeviceManager::tick(double elapsedTime) {
	for (auto it : mRenderPasses) it->tick(float(elapsedTime));
}

void DeviceManager::render() {
	if (!beginFrame()) return;
	for (auto it : mRenderPasses) it->beginFrame(mRenderContext.get());
	for (auto it : mRenderPasses) {
		if (!it->enabled()) continue;
		if (it->isCudaPass()) mRenderContext->beginCuda();
		else mRenderContext->endCuda();
		it->render(mRenderContext.get());
	}
	mRenderContext->endCuda();
	for (auto it : mRenderPasses) it->endFrame(mRenderContext.get());

	mCommandList->open();
	mHelperPass->BlitTexture(
		mCommandList, mSwapChainFramebuffers[getCurrentBackBufferIndex()],
		mRenderContext->getRenderTarget()->getColorTexture()->getTexture(),
							 mBindingCache.get());
	mCommandList->close();
	mNvrhiDevice->executeCommandList(mCommandList,
									  nvrhi::CommandQueue::Graphics);
}

void DeviceManager::updateAverageFrameTime(double elapsedTime) {
	mFrameTimeSum += elapsedTime;
	mNumberOfAccumulatedFrames += 1;

	if (mFrameTimeSum > mAverageTimeUpdateInterval && mNumberOfAccumulatedFrames > 0) {
		mAverageFrameTime			= mFrameTimeSum / double(mNumberOfAccumulatedFrames);
		mNumberOfAccumulatedFrames = 0;
		mFrameTimeSum				= 0.0;
	}
}

void DeviceManager::runMessageLoop() {
	glfwSetTime(0);
	mPreviousFrameTimestamp = glfwGetTime();

	while (!glfwWindowShouldClose(mWindow) && !gpContext->shouldQuit()) {

		if (mcallbacks.beforeFrame) mcallbacks.beforeFrame(*this);
		++mFrameIndex;		// so we denote the first frame as #1.
		glfwPollEvents();
		updateWindowSize();

		double curTime	   = glfwGetTime();
		double elapsedTime = curTime - mPreviousFrameTimestamp;

		if (mwindowVisible) {
			if (mcallbacks.beforeTick) mcallbacks.beforeTick(*this);
			tick(curTime);
			if (mcallbacks.afterTick) mcallbacks.afterTick(*this);
			if (mcallbacks.beforeRender) mcallbacks.beforeRender(*this);
			render();
			if (mcallbacks.afterRender) mcallbacks.afterRender(*this);
			if (mcallbacks.beforePresent) mcallbacks.beforePresent(*this);
			present();
			if (mcallbacks.afterPresent) mcallbacks.afterPresent(*this);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(0));

		getDevice()->runGarbageCollection();

		updateAverageFrameTime(elapsedTime);
		mPreviousFrameTimestamp = curTime;
	}

	getDevice()->waitForIdle();
}

Vector2i DeviceManager::getFrameSize() const {
	return {(int32_t) mDeviceParams.backBufferWidth,
			(int32_t) mDeviceParams.backBufferHeight};
}

void DeviceManager::getFrameSize(int &width, int &height) const {
	width  = mDeviceParams.backBufferWidth;
	height = mDeviceParams.backBufferHeight;
}

void DeviceManager::updateWindowSize() {
	if (mWindow == nullptr) return;
	int width, height;
	glfwGetFramebufferSize(mWindow, &width, &height);

	if (width == 0 || height == 0) {
		// window is minimized
		mwindowVisible = false;
		return;
	}

	mwindowVisible = true;

	if (int(mDeviceParams.backBufferWidth) != width ||
		int(mDeviceParams.backBufferHeight) != height ||
		(mDeviceParams.vsyncEnabled != mRequestedVSync)) {
		// window is not minimized, and the size has changed

		mNvrhiDevice->waitForIdle();
		backBufferResizing();

		mDeviceParams.backBufferWidth	= width;
		mDeviceParams.backBufferHeight = height;
		mDeviceParams.vsyncEnabled		= mRequestedVSync;

		resizeSwapChain();
		backBufferResized();
	}

	mDeviceParams.vsyncEnabled = mRequestedVSync;
}

void DeviceManager::onWindowPosUpdate(int x, int y) {
#ifdef _WINDOWS
	if (mDeviceParams.enablePerMonitorDPI) {
		HWND hwnd	 = glfwGetWin32Window(mWindow);
		auto monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);

		unsigned int dpiX;
		unsigned int dpiY;
		GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY);

		mDPIScaleFactorX = dpiX / 96.f;
		mDPIScaleFactorY = dpiY / 96.f;
	}
#endif
}

bool DeviceManager::onMouseEvent(io::MouseEvent &mouseEvent) {
	for (auto it = mRenderPasses.crbegin(); it != mRenderPasses.crend(); it++) {
		bool ret = (*it)->onMouseEvent(mouseEvent);
		if (ret) return true;
	}
	return false;
}

bool DeviceManager::onKeyEvent(io::KeyboardEvent &keyEvent) {
	for (auto it = mRenderPasses.crbegin(); it != mRenderPasses.crend(); it++) {
		bool ret = (*it)->onKeyEvent(keyEvent);
		if (ret) return true;
	}
	return false;
}

void DeviceManager::shutdown() {
	std::exception_ptr error;
	auto attempt = [&](auto operation) {
		try { operation(); }
		catch (...) { if (!error) error = std::current_exception(); }
	};
	if (mRenderContext) attempt([&] { mRenderContext->endCuda(); });
	if (mNvrhiDevice) {
		attempt([] { CUDA_CHECK(cudaDeviceSynchronize()); });
		attempt([&] {
			if (!mNvrhiDevice->waitForIdle()) throw std::runtime_error("Could not drain graphics work during shutdown.");
		});
	}
	mRenderPasses.clear();
	mSwapChainFramebuffers.clear();
	mCommandList = nullptr;
	mRenderContext.reset();
	mBindingCache.reset();
	mHelperPass.reset();

	destroyDeviceAndSwapChain();

	if (mWindow) {
		glfwDestroyWindow(mWindow);
		mWindow = nullptr;
	}

	if (mGlfwInitialized) {
		glfwTerminate();
		mGlfwInitialized = false;
	}
	if (error) std::rethrow_exception(error);
}

void DeviceManager::setWindowTitle(const char *title) {
	assert(title);
	if (mWindowTitle == title) return;

	if (mWindow) glfwSetWindowTitle(mWindow, title);

	mWindowTitle = title;
}

bool DeviceManager::createDeviceAndSwapChain() {
	switch (mDeviceParams.graphicsApi) {
	case nvrhi::GraphicsAPI::VULKAN:
		mBackend = createVulkanBackend();
		break;
#if KRR_ENABLE_D3D12
	case nvrhi::GraphicsAPI::D3D12:
		mBackend = createD3D12Backend();
		break;
#endif
	default:
		throw std::runtime_error("The requested graphics API is not enabled in this build.");
	}
	mBackend->initialize(mDeviceParams, mWindow, &DefaultMessageCallback::GetInstance());
	mNvrhiDevice = mBackend->getDevice();
	if (mDeviceParams.enableNvrhiValidationLayer)
		mNvrhiDevice = nvrhi::validation::createValidationLayer(mNvrhiDevice);
	mRendererString = mBackend->getRendererString();
	mCommandList = mNvrhiDevice->createCommandList();
	mRenderContext = std::make_shared<RenderContext>(getDevice(), mBackend->createInterop(getDevice()));
	if (!mHeadless) {
		mHelperPass = std::make_unique<CommonRenderPasses>(getDevice());
		mBindingCache = std::make_unique<BindingCache>(getDevice());
	}
	return true;
}

void DeviceManager::destroyDeviceAndSwapChain() {
	mFrameAcquired = false;
	mNvrhiDevice = nullptr;
	mBackend.reset();
	mRendererString.clear();
}

void DeviceManager::resizeSwapChain() { mBackend->resizeSwapChain(mDeviceParams); }
bool DeviceManager::beginFrame() {
	mFrameAcquired = mBackend->beginFrame();
	if (mFrameAcquired) return true;
	int width = 0, height = 0;
	glfwGetFramebufferSize(mWindow, &width, &height);
	if (!width || !height) return false;
	mNvrhiDevice->waitForIdle();
	backBufferResizing();
	mDeviceParams.backBufferWidth = width;
	mDeviceParams.backBufferHeight = height;
	resizeSwapChain();
	backBufferResized();
	mFrameAcquired = mBackend->beginFrame();
	return mFrameAcquired;
}
void DeviceManager::present() {
	if (mFrameAcquired) mBackend->present();
	mFrameAcquired = false;
}

nvrhi::ITexture *DeviceManager::getCurrentBackBuffer() const {
	return getBackBuffer(getCurrentBackBufferIndex());
}

nvrhi::ITexture *DeviceManager::getBackBuffer(size_t index) const {
	return mBackend ? mBackend->getBackBuffer(index) : nullptr;
}

nvrhi::ITexture *DeviceManager::getRenderImage() const {
	return mRenderContext->getRenderTarget()->getColorTexture()->getTexture();
}

size_t DeviceManager::getCurrentBackBufferIndex() const {
	return mBackend ? mBackend->getCurrentBackBufferIndex() : 0;
}

size_t DeviceManager::getBackBufferCount() const {
	return mBackend ? mBackend->getBackBufferCount() : 0;
}

NAMESPACE_END(krr)
