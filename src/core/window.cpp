#include <cstdio>
#include <queue>
#include <iomanip>
#include <thread>
#include <sstream>

#include "window.h"
#include "logger.h"
#include "device/context.h"

#include "render/profiler/profiler.h"

#ifdef _WINDOWS
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

KRR_NAMESPACE_BEGIN

namespace {
using namespace krr::io;

class ApiCallbacks {
public:
	static void windowSizeCallback(GLFWwindow *pGlfwWindow, int width, int height) {
		// We also get here in case the window was minimized, so we need to
		// ignore it
		if (width * height == 0) {
			return;
		}
		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
	}

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

	static void charInputCallback(GLFWwindow *pGlfwWindow, uint32_t input) {
		KeyboardEvent event;
		event.type		= KeyboardEvent::Type::Input;
		event.codepoint = input;

		DeviceManager *manager = (DeviceManager *) glfwGetWindowUserPointer(pGlfwWindow);
		if (manager != nullptr) {
			manager->onKeyEvent(event);
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

static std::vector<const char *> stringSetToVector(const std::unordered_set<std::string> &set) {
	std::vector<const char *> ret;
	for (const auto &s : set) {
		ret.push_back(s.c_str());
	}
	return ret;
}

static const struct {
	nvrhi::Format format;
	uint32_t redBits;
	uint32_t greenBits;
	uint32_t blueBits;
	uint32_t alphaBits;
	uint32_t depthBits;
	uint32_t stencilBits;
} formatInfo[] = {
	{ nvrhi::Format::UNKNOWN,            0,  0,  0,  0,  0,  0, },
    { nvrhi::Format::R8_UINT,            8,  0,  0,  0,  0,  0, },
    { nvrhi::Format::RG8_UINT,           8,  8,  0,  0,  0,  0, },
    { nvrhi::Format::RG8_UNORM,          8,  8,  0,  0,  0,  0, },
    { nvrhi::Format::R16_UINT,          16,  0,  0,  0,  0,  0, },
    { nvrhi::Format::R16_UNORM,         16,  0,  0,  0,  0,  0, },
    { nvrhi::Format::R16_FLOAT,         16,  0,  0,  0,  0,  0, },
    { nvrhi::Format::RGBA8_UNORM,        8,  8,  8,  8,  0,  0, },
    { nvrhi::Format::RGBA8_SNORM,        8,  8,  8,  8,  0,  0, },
    { nvrhi::Format::BGRA8_UNORM,        8,  8,  8,  8,  0,  0, },
    { nvrhi::Format::SRGBA8_UNORM,       8,  8,  8,  8,  0,  0, },
    { nvrhi::Format::SBGRA8_UNORM,       8,  8,  8,  8,  0,  0, },
    { nvrhi::Format::R10G10B10A2_UNORM, 10, 10, 10,  2,  0,  0, },
    { nvrhi::Format::R11G11B10_FLOAT,   11, 11, 10,  0,  0,  0, },
    { nvrhi::Format::RG16_UINT,         16, 16,  0,  0,  0,  0, },
    { nvrhi::Format::RG16_FLOAT,        16, 16,  0,  0,  0,  0, },
    { nvrhi::Format::R32_UINT,          32,  0,  0,  0,  0,  0, },
    { nvrhi::Format::R32_FLOAT,         32,  0,  0,  0,  0,  0, },
    { nvrhi::Format::RGBA16_FLOAT,      16, 16, 16, 16,  0,  0, },
    { nvrhi::Format::RGBA16_UNORM,      16, 16, 16, 16,  0,  0, },
    { nvrhi::Format::RGBA16_SNORM,      16, 16, 16, 16,  0,  0, },
    { nvrhi::Format::RG32_UINT,         32, 32,  0,  0,  0,  0, },
    { nvrhi::Format::RG32_FLOAT,        32, 32,  0,  0,  0,  0, },
    { nvrhi::Format::RGB32_UINT,        32, 32, 32,  0,  0,  0, },
    { nvrhi::Format::RGB32_FLOAT,       32, 32, 32,  0,  0,  0, },
    { nvrhi::Format::RGBA32_UINT,       32, 32, 32, 32,  0,  0, },
    { nvrhi::Format::RGBA32_FLOAT,      32, 32, 32, 32,  0,  0, }
};

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

	this->mDeviceParams = params;
	mRequestedVSync	 = params.vsyncEnabled;

	glfwSetErrorCallback(ApiCallbacks::errorCallback);

	glfwDefaultWindowHints();

	bool foundFormat = false;
	for (const auto &info : formatInfo) {
		if (info.format == params.swapChainFormat) {
			glfwWindowHint(GLFW_RED_BITS, info.redBits);
			glfwWindowHint(GLFW_GREEN_BITS, info.greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, info.blueBits);
			glfwWindowHint(GLFW_ALPHA_BITS, info.alphaBits);
			glfwWindowHint(GLFW_DEPTH_BITS, info.depthBits);
			glfwWindowHint(GLFW_STENCIL_BITS, info.stencilBits);
			foundFormat = true;
			break;
		}
	}

	assert(foundFormat);

	glfwWindowHint(GLFW_SAMPLES, params.swapChainSampleCount);
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
	//glfwSetCharCallback(mWindow, ApiCallbacks::charInputCallback);
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
	beginFrame();
	for (auto it : mRenderPasses) it->beginFrame(mRenderContext.get());
	for (auto it : mRenderPasses) {
		if (!it->enabled()) continue;
		if (it->isCudaPass()) mRenderContext->sychronizeCuda();
		it->render(mRenderContext.get());
		if (it->isCudaPass()) mRenderContext->sychronizeVulkan();
	}
	for (auto it : mRenderPasses) it->endFrame(mRenderContext.get());

	mNvrhiDevice->queueSignalSemaphore(nvrhi::CommandQueue::Graphics,
										mPresentSemaphore, 0);
	mCommandList->open(); 
	mHelperPass->BlitTexture(
		mCommandList, mSwapChainFramebuffers[mSwapChainIndex],
		mRenderContext->getRenderTarget()->getColorTexture()->getVulkanTexture(),
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
	glfwGetWindowSize(mWindow, &width, &height);

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

static void ApplyDeadZone(Vector2f &v, const float deadZone = 0.1f) {
	v *= std::max(length(v) - deadZone, 0.f) / (1.f - deadZone);
}

void DeviceManager::shutdown() {
	mSwapChainFramebuffers.clear();
	mRenderContext.reset();
	mBindingCache.reset();
	mHelperPass.reset();

	destroyDeviceAndSwapChain();

	if (mWindow) {
		glfwDestroyWindow(mWindow);
		mWindow = nullptr;
	}

	glfwTerminate();
}

void DeviceManager::setWindowTitle(const char *title) {
	assert(title);
	if (mWindowTitle == title) return;

	glfwSetWindowTitle(mWindow, title);

	mWindowTitle = title;
}

template <typename T> static std::vector<T> setToVector(const std::unordered_set<T> &set) {
	std::vector<T> ret;
	for (const auto &s : set) {
		ret.push_back(s);
	}

	return ret;
}

bool DeviceManager::createInstance() {
	if (!glfwVulkanSupported()) {
		return false;
	}

	// add any extensions required by GLFW
	uint32_t glfwExtCount;
	const char **glfwExt = glfwGetRequiredInstanceExtensions(&glfwExtCount);
	assert(glfwExt);

	for (uint32_t i = 0; i < glfwExtCount; i++) {
		enabledExtensions.instance.insert(std::string(glfwExt[i]));
	}

	// add instance extensions requested by the user
	for (const std::string &name : mDeviceParams.requiredVulkanInstanceExtensions) {
		enabledExtensions.instance.insert(name);
	}
	for (const std::string &name : mDeviceParams.optionalVulkanInstanceExtensions) {
		optionalExtensions.instance.insert(name);
	}

	// add layers requested by the user
	for (const std::string &name : mDeviceParams.requiredVulkanLayers) {
		enabledExtensions.layers.insert(name);
	}
	for (const std::string &name : mDeviceParams.optionalVulkanLayers) {
		optionalExtensions.layers.insert(name);
	}

	std::unordered_set<std::string> requiredExtensions = enabledExtensions.instance;

	// figure out which optional extensions are supported
	for (const auto &instanceExt : vk::enumerateInstanceExtensionProperties()) {
		const std::string name = instanceExt.extensionName;
		if (optionalExtensions.instance.count(name)) 
			enabledExtensions.instance.insert(name);
		if (mCudaInteropExtensions.instance.count(name)) 
			enabledExtensions.instance.insert(name);
		requiredExtensions.erase(name);
	}

	if (!requiredExtensions.empty()) {
		std::stringstream ss;
		ss << "Cannot create a Vulkan instance because the following required extension(s) are not "
			  "supported:";
		for (const auto &ext : requiredExtensions) ss << std::endl << "  - " << ext;

		Log(Error, "%s", ss.str().c_str());
		return false;
	}

	logMessage(mDeviceParams.infoLogSeverity, "Enabled Vulkan instance extensions:");
	for (const auto &ext : enabledExtensions.instance) {
		Log(Info, "%s", ext.c_str());
	}

	std::unordered_set<std::string> requiredLayers = enabledExtensions.layers;

	for (const auto &layer : vk::enumerateInstanceLayerProperties()) {
		const std::string name = layer.layerName;
		if (optionalExtensions.layers.find(name) != optionalExtensions.layers.end()) {
			enabledExtensions.layers.insert(name);
		}

		requiredLayers.erase(name);
	}

	if (!requiredLayers.empty()) {
		std::stringstream ss;
		ss << "Cannot create a Vulkan instance because the following required layer(s) are not "
			  "supported:";
		for (const auto &ext : requiredLayers) ss << std::endl << "  - " << ext;

		Log(Error, "%s", ss.str().c_str());
		return false;
	}

	logMessage(mDeviceParams.infoLogSeverity, "Enabled Vulkan layers:");
	for (const auto &layer : enabledExtensions.layers) {
		Log(Info, "%s", layer.c_str());
	}

	auto instanceExtVec = stringSetToVector(enabledExtensions.instance);
	auto layerVec		= stringSetToVector(enabledExtensions.layers);

	auto applicationInfo = vk::ApplicationInfo().setApiVersion(VK_MAKE_VERSION(1, 2, 0));

	// create the vulkan instance
	vk::InstanceCreateInfo info = vk::InstanceCreateInfo()
									  .setEnabledLayerCount(uint32_t(layerVec.size()))
									  .setPpEnabledLayerNames(layerVec.data())
									  .setEnabledExtensionCount(uint32_t(instanceExtVec.size()))
									  .setPpEnabledExtensionNames(instanceExtVec.data())
									  .setPApplicationInfo(&applicationInfo);

	const vk::Result res = vk::createInstance(&info, nullptr, &mVulkanInstance);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan instance, error code = %s",
			nvrhi::vulkan::resultToString(res));
		return false;
	}

	VULKAN_HPP_DEFAULT_DISPATCHER.init(mVulkanInstance);

	return true;
}

void DeviceManager::installDebugCallback() {
	auto info =
		vk::DebugReportCallbackCreateInfoEXT()
			.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning |
					  //   vk::DebugReportFlagBitsEXT::eInformation |
					  vk::DebugReportFlagBitsEXT::ePerformanceWarning)
			.setPfnCallback(vulkanDebugCallback)
			.setPUserData(this);

	vk::Result res =
		mVulkanInstance.createDebugReportCallbackEXT(&info, nullptr, &mDebugReportCallback);
	assert(res == vk::Result::eSuccess);
}

bool DeviceManager::pickPhysicalDevice() {
	vk::Format requestedFormat = nvrhi::vulkan::convertFormat(mDeviceParams.swapChainFormat);
	vk::Extent2D requestedExtent(mDeviceParams.backBufferWidth, mDeviceParams.backBufferHeight);

	auto devices = mVulkanInstance.enumeratePhysicalDevices();

	// Start building an error message in case we cannot find a device.
	std::stringstream errorStream;
	errorStream
		<< "Cannot find a Vulkan device that supports all the required extensions and properties.";

	// build a list of GPUs
	std::vector<vk::PhysicalDevice> discreteGPUs;
	std::vector<vk::PhysicalDevice> otherGPUs;
	for (const auto &dev : devices) {
		auto prop = dev.getProperties();

		errorStream << std::endl << prop.deviceName.data() << ":";

		// check that all required device extensions are present
		std::unordered_set<std::string> requiredExtensions = enabledExtensions.device;
		auto deviceExtensions = dev.enumerateDeviceExtensionProperties();
		for (const auto &ext : deviceExtensions) {
			requiredExtensions.erase(std::string(ext.extensionName.data()));
		}

		bool deviceIsGood = true;

		if (!requiredExtensions.empty()) {
			// device is missing one or more required extensions
			for (const auto &ext : requiredExtensions) {
				errorStream << std::endl << "  - missing " << ext;
			}
			deviceIsGood = false;
		}

		auto deviceFeatures = dev.getFeatures();
		if (!deviceFeatures.samplerAnisotropy) {
			// device is a toaster oven
			errorStream << std::endl << "  - does not support samplerAnisotropy";
			deviceIsGood = false;
		}
		if (!deviceFeatures.textureCompressionBC) {
			errorStream << std::endl << "  - does not support textureCompressionBC";
			deviceIsGood = false;
		}

		// check that this device supports our intended swap chain creation parameters
		auto surfaceCaps   = dev.getSurfaceCapabilitiesKHR(mWindowSurface);
		auto surfaceFmts   = dev.getSurfaceFormatsKHR(mWindowSurface);
		auto surfacePModes = dev.getSurfacePresentModesKHR(mWindowSurface);

		if (surfaceCaps.minImageCount > mDeviceParams.swapChainBufferCount ||
			(surfaceCaps.maxImageCount < mDeviceParams.swapChainBufferCount &&
			 surfaceCaps.maxImageCount > 0)) {
			errorStream << std::endl << "  - cannot support the requested swap chain image count:";
			errorStream << " requested " << mDeviceParams.swapChainBufferCount << ", available "
						<< surfaceCaps.minImageCount << " - " << surfaceCaps.maxImageCount;
			deviceIsGood = false;
		}

		if (surfaceCaps.minImageExtent.width > requestedExtent.width ||
			surfaceCaps.minImageExtent.height > requestedExtent.height ||
			surfaceCaps.maxImageExtent.width < requestedExtent.width ||
			surfaceCaps.maxImageExtent.height < requestedExtent.height) {
			errorStream << std::endl << "  - cannot support the requested swap chain size:";
			errorStream << " requested " << requestedExtent.width << "x" << requestedExtent.height
						<< ", ";
			errorStream << " available " << surfaceCaps.minImageExtent.width << "x"
						<< surfaceCaps.minImageExtent.height;
			errorStream << " - " << surfaceCaps.maxImageExtent.width << "x"
						<< surfaceCaps.maxImageExtent.height;
			deviceIsGood = false;
		}

		bool surfaceFormatPresent = false;
		for (const vk::SurfaceFormatKHR &surfaceFmt : surfaceFmts) {
			if (surfaceFmt.format == requestedFormat) {
				surfaceFormatPresent = true;
				break;
			}
		}

		if (!surfaceFormatPresent) {
			// can't create a swap chain using the format requested
			errorStream << std::endl << "  - does not support the requested swap chain format";
			deviceIsGood = false;
		}

		if (!findQueueFamilies(dev)) {
			// device doesn't have all the queue families we need
			errorStream << std::endl << "  - does not support the necessary queue types";
			deviceIsGood = false;
		}

		// check that we can present from the graphics queue
		uint32_t canPresent = dev.getSurfaceSupportKHR(mGraphicsQueueFamily, mWindowSurface);
		if (!canPresent) {
			errorStream << std::endl << "  - cannot present";
			deviceIsGood = false;
		}

		if (!deviceIsGood) continue;

		if (prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			discreteGPUs.push_back(dev);
		} else {
			otherGPUs.push_back(dev);
		}
	}

	// pick the first discrete GPU if it exists, otherwise the first integrated GPU
	if (!discreteGPUs.empty()) {
		mVulkanPhysicalDevice = discreteGPUs[0];
		return true;
	}

	if (!otherGPUs.empty()) {
		mVulkanPhysicalDevice = otherGPUs[0];
		return true;
	}

	Log(Error, "%s", errorStream.str().c_str());

	return false;
}

bool DeviceManager::findQueueFamilies(vk::PhysicalDevice physicalDevice) {
	auto props = physicalDevice.getQueueFamilyProperties();

	for (int i = 0; i < int(props.size()); i++) {
		const auto &queueFamily = props[i];

		if (mGraphicsQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				mGraphicsQueueFamily = i;
			}
		}

		if (mComputeQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				mComputeQueueFamily = i;
			}
		}

		if (mTransferQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				mTransferQueueFamily = i;
			}
		}

		if (mPresentQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				glfwGetPhysicalDevicePresentationSupport(mVulkanInstance, physicalDevice, i)) {
				mPresentQueueFamily = i;
			}
		}
	}

	if (mGraphicsQueueFamily == -1 || mPresentQueueFamily == -1 ||
		(mComputeQueueFamily == -1 && mDeviceParams.enableComputeQueue) ||
		(mTransferQueueFamily == -1 && mDeviceParams.enableCopyQueue)) {
		return false;
	}

	return true;
}

bool DeviceManager::createDevice() {
	// figure out which optional extensions are supported
	auto deviceExtensions = mVulkanPhysicalDevice.enumerateDeviceExtensionProperties();
	for (const auto &ext : deviceExtensions) {
		const std::string name = ext.extensionName;
		if (optionalExtensions.device.count(name)) 
			enabledExtensions.device.insert(name);
		if (mCudaInteropExtensions.device.count(name)) 
			enabledExtensions.device.insert(name);
		if (mDeviceParams.enableRayTracingExtensions && mRayTracingExtensions.count(name))
			enabledExtensions.device.insert(name);
	}

	bool accelStructSupported	= false;
	bool bufferAddressSupported = false;
	bool rayPipelineSupported	= false;
	bool rayQuerySupported		= false;
	bool meshletsSupported		= false;
	bool vrsSupported			= false;

	logMessage(mDeviceParams.infoLogSeverity, "Enabled Vulkan device extensions:");
	for (const auto &ext : enabledExtensions.device) {
		Log(Info, "%s", ext.c_str());

		if (ext == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
			accelStructSupported = true;
		else if (ext == VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
			bufferAddressSupported = true;
		else if (ext == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
			rayPipelineSupported = true;
		else if (ext == VK_KHR_RAY_QUERY_EXTENSION_NAME)
			rayQuerySupported = true;
		else if (ext == VK_NV_MESH_SHADER_EXTENSION_NAME)
			meshletsSupported = true;
		else if (ext == VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)
			vrsSupported = true;
	}

	std::unordered_set<int> uniqueQueueFamilies = {mGraphicsQueueFamily, mPresentQueueFamily};

	if (mDeviceParams.enableComputeQueue) uniqueQueueFamilies.insert(mComputeQueueFamily);

	if (mDeviceParams.enableCopyQueue) uniqueQueueFamilies.insert(mTransferQueueFamily);

	float priority = 1.f;
	std::vector<vk::DeviceQueueCreateInfo> queueDesc;
	for (int queueFamily : uniqueQueueFamilies) {
		queueDesc.push_back(vk::DeviceQueueCreateInfo()
								.setQueueFamilyIndex(queueFamily)
								.setQueueCount(1)
								.setPQueuePriorities(&priority));
	}

	auto accelStructFeatures =
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR().setAccelerationStructure(true);
	auto bufferAddressFeatures =
		vk::PhysicalDeviceBufferAddressFeaturesEXT().setBufferDeviceAddress(true);
	auto rayPipelineFeatures = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR()
								   .setRayTracingPipeline(true)
								   .setRayTraversalPrimitiveCulling(true);
	auto rayQueryFeatures = vk::PhysicalDeviceRayQueryFeaturesKHR().setRayQuery(true);
	auto meshletFeatures =
		vk::PhysicalDeviceMeshShaderFeaturesNV().setTaskShader(true).setMeshShader(true);
	auto vrsFeatures = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR()
						   .setPipelineFragmentShadingRate(true)
						   .setPrimitiveFragmentShadingRate(true)
						   .setAttachmentFragmentShadingRate(true);

	void *pNext = nullptr;
#define APPEND_EXTENSION(condition, desc)                                                          \
	if (condition) {                                                                               \
		(desc).pNext = pNext;                                                                      \
		pNext		 = &(desc);                                                                    \
	} // NOLINT(cppcoreguidelines-macro-usage)
	APPEND_EXTENSION(accelStructSupported, accelStructFeatures)
	APPEND_EXTENSION(bufferAddressSupported, bufferAddressFeatures)
	APPEND_EXTENSION(rayPipelineSupported, rayPipelineFeatures)
	APPEND_EXTENSION(rayQuerySupported, rayQueryFeatures)
	APPEND_EXTENSION(meshletsSupported, meshletFeatures)
	APPEND_EXTENSION(vrsSupported, vrsFeatures)
#undef APPEND_EXTENSION

	auto deviceFeatures = vk::PhysicalDeviceFeatures()
							  .setShaderImageGatherExtended(true)
							  .setSamplerAnisotropy(true)
							  .setTessellationShader(true)
							  .setTextureCompressionBC(true)
							  .setGeometryShader(true)
							  .setImageCubeArray(true)
							  .setDualSrcBlend(true);

	auto vulkan12features = vk::PhysicalDeviceVulkan12Features()
								.setDescriptorIndexing(true)
								.setRuntimeDescriptorArray(true)
								.setDescriptorBindingPartiallyBound(true)
								.setDescriptorBindingVariableDescriptorCount(true)
								.setTimelineSemaphore(true)
								.setShaderSampledImageArrayNonUniformIndexing(true)
								.setBufferDeviceAddress(bufferAddressSupported)
								.setPNext(pNext);

	auto layerVec = stringSetToVector(enabledExtensions.layers);
	auto extVec	  = stringSetToVector(enabledExtensions.device);

	auto deviceDesc = vk::DeviceCreateInfo()
						  .setPQueueCreateInfos(queueDesc.data())
						  .setQueueCreateInfoCount(uint32_t(queueDesc.size()))
						  .setPEnabledFeatures(&deviceFeatures)
						  .setEnabledExtensionCount(uint32_t(extVec.size()))
						  .setPpEnabledExtensionNames(extVec.data())
						  .setEnabledLayerCount(uint32_t(layerVec.size()))
						  .setPpEnabledLayerNames(layerVec.data())
						  .setPNext(&vulkan12features);

	if (mDeviceParams.deviceCreateInfoCallback)
		mDeviceParams.deviceCreateInfoCallback(deviceDesc);

	const vk::Result res =
		mVulkanPhysicalDevice.createDevice(&deviceDesc, nullptr, &mVulkanDevice);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan physical device, error code = %s",
			nvrhi::vulkan::resultToString(res));
		return false;
	}

	mVulkanDevice.getQueue(mGraphicsQueueFamily, 0, &mGraphicsQueue);
	if (mDeviceParams.enableComputeQueue)
		mVulkanDevice.getQueue(mComputeQueueFamily, 0, &mComputeQueue);
	if (mDeviceParams.enableCopyQueue)
		mVulkanDevice.getQueue(mTransferQueueFamily, 0, &mTransferQueue);
	mVulkanDevice.getQueue(mPresentQueueFamily, 0, &mPresentQueue);

	VULKAN_HPP_DEFAULT_DISPATCHER.init(mVulkanDevice);

	// stash the renderer string
	auto prop		 = mVulkanPhysicalDevice.getProperties();
	mRendererString = std::string(prop.deviceName.data());

	Log(Info, "Created Vulkan device: %s", mRendererString.c_str());
	return true;
}

bool DeviceManager::createWindowSurface() {
	const VkResult res = glfwCreateWindowSurface(mVulkanInstance, mWindow, nullptr,
												 (VkSurfaceKHR *) &mWindowSurface);
	if (res != VK_SUCCESS) {
		Log(Error, "Failed to create a GLFW window surface, error code = %s",
			nvrhi::vulkan::resultToString(res));
		return false;
	}

	return true;
}

void DeviceManager::destroySwapChain() {
	if (mVulkanDevice) {
		mVulkanDevice.waitIdle();
	}

	if (mSwapChain) {
		mVulkanDevice.destroySwapchainKHR(mSwapChain);
		mSwapChain = nullptr;
	}

	mSwapChainImages.clear();
}

// This routine will be called whenever resizing...
bool DeviceManager::createSwapChain() {
	destroySwapChain();

	mSwapChainFormat = {vk::Format(nvrhi::vulkan::convertFormat(mDeviceParams.swapChainFormat)),
						 vk::ColorSpaceKHR::eSrgbNonlinear};

	vk::Extent2D extent =
		vk::Extent2D(mDeviceParams.backBufferWidth, mDeviceParams.backBufferHeight);

	std::unordered_set<uint32_t> uniqueQueues = {uint32_t(mGraphicsQueueFamily),
												 uint32_t(mPresentQueueFamily)};

	std::vector<uint32_t> queues = setToVector(uniqueQueues);

	const bool enableSwapChainSharing = queues.size() > 1;

	auto desc =
		vk::SwapchainCreateInfoKHR()
			.setSurface(mWindowSurface)
			.setMinImageCount(mDeviceParams.swapChainBufferCount)
			.setImageFormat(mSwapChainFormat.format)
			.setImageColorSpace(mSwapChainFormat.colorSpace)
			.setImageExtent(extent)
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment |
						   vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
			.setImageSharingMode(enableSwapChainSharing ? vk::SharingMode::eConcurrent
														: vk::SharingMode::eExclusive)
			.setQueueFamilyIndexCount(enableSwapChainSharing ? uint32_t(queues.size()) : 0)
			.setPQueueFamilyIndices(enableSwapChainSharing ? queues.data() : nullptr)
			.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity)
			.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
			.setPresentMode(mDeviceParams.vsyncEnabled ? vk::PresentModeKHR::eFifo
														: vk::PresentModeKHR::eImmediate)
			.setClipped(true)
			.setOldSwapchain(nullptr);

	const vk::Result res = mVulkanDevice.createSwapchainKHR(&desc, nullptr, &mSwapChain);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan swap chain, error code = %s",
			nvrhi::vulkan::resultToString(res));
		return false;
	}

	// retrieve swap chain images
	auto images = mVulkanDevice.getSwapchainImagesKHR(mSwapChain);
	for (auto image : images) {
		SwapChainImage sci;
		sci.image = image;

		nvrhi::TextureDesc textureDesc;
		textureDesc.width			 = mDeviceParams.backBufferWidth;
		textureDesc.height			 = mDeviceParams.backBufferHeight;
		textureDesc.format			 = mDeviceParams.swapChainFormat;
		textureDesc.debugName		 = "Swap chain image";
		textureDesc.initialState	 = nvrhi::ResourceStates::Present;
		textureDesc.keepInitialState = true;
		textureDesc.isRenderTarget	 = true;

		sci.rhiHandle = mNvrhiDevice->createHandleForNativeTexture(
			nvrhi::ObjectTypes::VK_Image, nvrhi::Object(sci.image), textureDesc);
		mSwapChainImages.push_back(sci);
	}
	mSwapChainIndex = 0;
	return true;
}

bool DeviceManager::createDeviceAndSwapChain() {
	if (mDeviceParams.enableDebugRuntime) {
		enabledExtensions.instance.insert("VK_EXT_debug_report");
		enabledExtensions.layers.insert("VK_LAYER_KHRONOS_validation");
	}

	const vk::DynamicLoader dl;
	const PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = // NOLINT(misc-misplaced-const)
		dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(a)                                                                                   \
	if (!(a)) {                                                                                    \
		return false;                                                                              \
	}
	CHECK(createInstance())

	if (mDeviceParams.enableDebugRuntime) {
		installDebugCallback();
	}

	if (mDeviceParams.swapChainFormat == nvrhi::Format::SRGBA8_UNORM)
		mDeviceParams.swapChainFormat = nvrhi::Format::SBGRA8_UNORM;
	else if (mDeviceParams.swapChainFormat == nvrhi::Format::RGBA8_UNORM)
		mDeviceParams.swapChainFormat = nvrhi::Format::BGRA8_UNORM;

	// add device extensions requested by the user
	for (const std::string &name : mDeviceParams.requiredVulkanDeviceExtensions) {
		enabledExtensions.device.insert(name);
	}
	for (const std::string &name : mDeviceParams.optionalVulkanDeviceExtensions) {
		optionalExtensions.device.insert(name);
	}

	CHECK(createWindowSurface())
	CHECK(pickPhysicalDevice())
	CHECK(findQueueFamilies(mVulkanPhysicalDevice))
	CHECK(createDevice())

	auto vecInstanceExt = stringSetToVector(enabledExtensions.instance);
	auto vecLayers		= stringSetToVector(enabledExtensions.layers);
	auto vecDeviceExt	= stringSetToVector(enabledExtensions.device);

	nvrhi::vulkan::DeviceDesc deviceDesc;
	deviceDesc.errorCB			  = &DefaultMessageCallback::GetInstance();
	deviceDesc.instance			  = mVulkanInstance;
	deviceDesc.physicalDevice	  = mVulkanPhysicalDevice;
	deviceDesc.device			  = mVulkanDevice;
	deviceDesc.graphicsQueue	  = mGraphicsQueue;
	deviceDesc.graphicsQueueIndex = mGraphicsQueueFamily;
	if (mDeviceParams.enableComputeQueue) {
		deviceDesc.computeQueue		 = mComputeQueue;
		deviceDesc.computeQueueIndex = mComputeQueueFamily;
	}
	if (mDeviceParams.enableCopyQueue) {
		deviceDesc.transferQueue	  = mTransferQueue;
		deviceDesc.transferQueueIndex = mTransferQueueFamily;
	}
	deviceDesc.instanceExtensions	 = vecInstanceExt.data();
	deviceDesc.numInstanceExtensions = vecInstanceExt.size();
	deviceDesc.deviceExtensions		 = vecDeviceExt.data();
	deviceDesc.numDeviceExtensions	 = vecDeviceExt.size();

	mNvrhiDevice = nvrhi::vulkan::createDevice(deviceDesc);
	gpContext->setDefaultVkDevice(mNvrhiDevice.Get());

	if (mDeviceParams.enableNvrhiValidationLayer) {
		mValidationLayer = nvrhi::validation::createValidationLayer(mNvrhiDevice);
	}

	mCudaHandler = std::make_unique<vkrhi::CuVkHandler>(getDevice());
	mCudaHandler->initCUDA();

	vkrhi::CommandListParameters;
	mPresentSemaphore = mCudaHandler->createCuVkSemaphore(false); 
	// [not that descent] to make cuda wait for previous vulkan operations, 
	// I use the following dirty routines to replace the queue semaphore with a
	// vulkan-exported semaphore.

	mCommandList   = mNvrhiDevice->createCommandList();
	mRenderContext = std::make_shared<RenderContext>(getDevice());
	mHelperPass	   = std::make_unique<CommonRenderPasses>(getDevice());
	mBindingCache  = std::make_unique<BindingCache>(getDevice());

	auto *graphicsQueue = dynamic_cast<vkrhi::vulkan::Device *>(mNvrhiDevice.Get())
							->getQueue(nvrhi::CommandQueue::Graphics);
	mVulkanDevice.waitIdle();
	mVulkanDevice.destroySemaphore(graphicsQueue->trackingSemaphore);
	graphicsQueue->trackingSemaphore = mRenderContext->getVulkanSemaphore();
	CHECK(createSwapChain())

#undef CHECK
	return true;
}

void DeviceManager::destroyDeviceAndSwapChain() {
	destroySwapChain();
	mVulkanDevice.waitIdle();
	mVulkanDevice.destroySemaphore(mPresentSemaphore);
	mPresentSemaphore = {};

	mCommandList = nullptr;

	mNvrhiDevice	  = nullptr;
	mValidationLayer = nullptr;
	mRendererString.clear();
	gpContext->setDefaultVkDevice(nullptr);

	if (mDebugReportCallback) {
		mVulkanInstance.destroyDebugReportCallbackEXT(mDebugReportCallback);
	}

	if (mVulkanDevice) {
		mVulkanDevice.destroy();
		mVulkanDevice = nullptr;
	}

	if (mWindowSurface) {
		assert(mVulkanInstance);
		mVulkanInstance.destroySurfaceKHR(mWindowSurface);
		mWindowSurface = nullptr;
	}

	if (mVulkanInstance) {
		mVulkanInstance.destroy();
		mVulkanInstance = nullptr;
	}
}

void DeviceManager::beginFrame() {
	const vk::Result res =
		mVulkanDevice.acquireNextImageKHR(mSwapChain,
										   std::numeric_limits<uint64_t>::max(), // timeout
										   mPresentSemaphore, vk::Fence(), &mSwapChainIndex);

	assert(res == vk::Result::eSuccess);

	mNvrhiDevice->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, mPresentSemaphore, 0);
}

void DeviceManager::present() {

	vk::PresentInfoKHR info = vk::PresentInfoKHR()
								  .setWaitSemaphoreCount(1)
								  .setPWaitSemaphores(&mPresentSemaphore.vulkan())
								  .setSwapchainCount(1)
								  .setPSwapchains(&mSwapChain)
								  .setPImageIndices(&mSwapChainIndex);

	const vk::Result res = mPresentQueue.presentKHR(&info);
	assert(res == vk::Result::eSuccess || res == vk::Result::eErrorOutOfDateKHR);

	if (mDeviceParams.enableDebugRuntime) {
		// according to vulkan-tutorial.com, "the validation layer implementation expects
		// the application to explicitly synchronize with the GPU"
		mPresentQueue.waitIdle();
	} else {
#ifndef _WIN32
		if (mDeviceParams.vsyncEnabled) {
			mPresentQueue.waitIdle();
		}
#endif

		while (mFramesInFlight.size() > mDeviceParams.maxFramesInFlight) {
			auto query = mFramesInFlight.front();
			mFramesInFlight.pop();

			mNvrhiDevice->waitEventQuery(query);

			mQueryPool.push_back(query);
		}

		nvrhi::EventQueryHandle query;
		if (!mQueryPool.empty()) {
			query = mQueryPool.back();
			mQueryPool.pop_back();
		} else {
			query = mNvrhiDevice->createEventQuery();
		}

		mNvrhiDevice->resetEventQuery(query);
		mNvrhiDevice->setEventQuery(query, nvrhi::CommandQueue::Graphics);
		mFramesInFlight.push(query);
	}
}

KRR_NAMESPACE_END