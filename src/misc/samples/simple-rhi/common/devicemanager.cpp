#include "DeviceManager.h"

#include <cstdio>
#include <queue>
#include <iomanip>
#include <thread>
#include <sstream>

#ifdef _WINDOWS
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")
#endif

// Define the Vulkan dynamic dispatcher 
// this needs to occur in exactly one cpp file in the program.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

KRR_NAMESPACE_BEGIN

static void ErrorCallback_GLFW(int error, const char *description) {
	fprintf(stderr, "GLFW error: %s\n", description);
	exit(1);
}

static void WindowIconifyCallback_GLFW(GLFWwindow *window, int iconified) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->WindowIconifyCallback(iconified);
}

static void WindowFocusCallback_GLFW(GLFWwindow *window, int focused) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->WindowFocusCallback(focused);
}

static void WindowRefreshCallback_GLFW(GLFWwindow *window) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->WindowRefreshCallback();
}

static void WindowCloseCallback_GLFW(GLFWwindow *window) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->WindowCloseCallback();
}

static void WindowPosCallback_GLFW(GLFWwindow *window, int xpos, int ypos) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->WindowPosCallback(xpos, ypos);
}

static void KeyCallback_GLFW(GLFWwindow *window, int key, int scancode, int action, int mods) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->KeyboardUpdate(key, scancode, action, mods);
}

static void CharModsCallback_GLFW(GLFWwindow *window, unsigned int unicode, int mods) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->KeyboardCharInput(unicode, mods);
}

static void MousePosCallback_GLFW(GLFWwindow *window, double xpos, double ypos) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->MousePosUpdate(xpos, ypos);
}

static void MouseButtonCallback_GLFW(GLFWwindow *window, int button, int action, int mods) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->MouseButtonUpdate(button, action, mods);
}

static void MouseScrollCallback_GLFW(GLFWwindow *window, double xoffset, double yoffset) {
	DeviceManager *manager = reinterpret_cast<DeviceManager *>(glfwGetWindowUserPointer(window));
	manager->MouseScrollUpdate(xoffset, yoffset);
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

bool DeviceManager::CreateWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
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

	this->m_DeviceParams = params;
	m_RequestedVSync	 = params.vsyncEnabled;

	glfwSetErrorCallback(ErrorCallback_GLFW);

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

	m_Window = glfwCreateWindow(
		params.backBufferWidth, params.backBufferHeight, windowTitle ? windowTitle : "",
		params.startFullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

	if (m_Window == nullptr) {
		return false;
	}

	if (params.startFullscreen) {
		glfwSetWindowMonitor(m_Window, glfwGetPrimaryMonitor(), 0, 0,
							 m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight,
							 m_DeviceParams.refreshRate);
	} else {
		int fbWidth = 0, fbHeight = 0;
		glfwGetFramebufferSize(m_Window, &fbWidth, &fbHeight);
		m_DeviceParams.backBufferWidth	= fbWidth;
		m_DeviceParams.backBufferHeight = fbHeight;
	}

	if (windowTitle) m_WindowTitle = windowTitle;

	glfwSetWindowUserPointer(m_Window, this);

	if (params.windowPosX != -1 && params.windowPosY != -1) {
		glfwSetWindowPos(m_Window, params.windowPosX, params.windowPosY);
	}

	if (params.startMaximized) {
		glfwMaximizeWindow(m_Window);
	}

	glfwSetWindowPosCallback(m_Window, WindowPosCallback_GLFW);
	glfwSetWindowCloseCallback(m_Window, WindowCloseCallback_GLFW);
	glfwSetWindowRefreshCallback(m_Window, WindowRefreshCallback_GLFW);
	glfwSetWindowFocusCallback(m_Window, WindowFocusCallback_GLFW);
	glfwSetWindowIconifyCallback(m_Window, WindowIconifyCallback_GLFW);
	glfwSetKeyCallback(m_Window, KeyCallback_GLFW);
	glfwSetCharModsCallback(m_Window, CharModsCallback_GLFW);
	glfwSetCursorPosCallback(m_Window, MousePosCallback_GLFW);
	glfwSetMouseButtonCallback(m_Window, MouseButtonCallback_GLFW);
	glfwSetScrollCallback(m_Window, MouseScrollCallback_GLFW);

	if (!CreateDeviceAndSwapChain()) return false;

	glfwShowWindow(m_Window);

	// reset the back buffer size state to enforce a resize event
	m_DeviceParams.backBufferWidth	= 0;
	m_DeviceParams.backBufferHeight = 0;

	UpdateWindowSize();

	return true;
}

void DeviceManager::AddRenderPassToFront(IRenderPass *pRenderPass) {
	m_vRenderPasses.remove(pRenderPass);
	m_vRenderPasses.push_front(pRenderPass);

	pRenderPass->BackBufferResizing();
	pRenderPass->BackBufferResized(m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight,
								   m_DeviceParams.swapChainSampleCount);
}

void DeviceManager::AddRenderPassToBack(IRenderPass *pRenderPass) {
	m_vRenderPasses.remove(pRenderPass);
	m_vRenderPasses.push_back(pRenderPass);

	pRenderPass->BackBufferResizing();
	pRenderPass->BackBufferResized(m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight,
								   m_DeviceParams.swapChainSampleCount);
}

void DeviceManager::RemoveRenderPass(IRenderPass *pRenderPass) {
	m_vRenderPasses.remove(pRenderPass);
}

void DeviceManager::BackBufferResizing() {
	m_SwapChainFramebuffers.clear();

	for (auto it : m_vRenderPasses) {
		it->BackBufferResizing();
	}
}

void DeviceManager::BackBufferResized() {
	for (auto it : m_vRenderPasses) {
		it->BackBufferResized(m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight,
							  m_DeviceParams.swapChainSampleCount);
	}

	uint32_t backBufferCount = GetBackBufferCount();
	m_SwapChainFramebuffers.resize(backBufferCount);
	for (uint32_t index = 0; index < backBufferCount; index++) {
		m_SwapChainFramebuffers[index] = GetDevice()->createFramebuffer(
			nvrhi::FramebufferDesc().addColorAttachment(GetBackBuffer(index)));
	}
}

void DeviceManager::Animate(double elapsedTime) {
	for (auto it : m_vRenderPasses) {
		it->Animate(float(elapsedTime));
	}
}

void DeviceManager::Render() {
	BeginFrame();

	nvrhi::IFramebuffer *framebuffer = m_SwapChainFramebuffers[GetCurrentBackBufferIndex()];

	for (auto it : m_vRenderPasses) {
		it->Render(framebuffer);
	}
}

void DeviceManager::UpdateAverageFrameTime(double elapsedTime) {
	m_FrameTimeSum += elapsedTime;
	m_NumberOfAccumulatedFrames += 1;

	if (m_FrameTimeSum > m_AverageTimeUpdateInterval && m_NumberOfAccumulatedFrames > 0) {
		m_AverageFrameTime			= m_FrameTimeSum / double(m_NumberOfAccumulatedFrames);
		m_NumberOfAccumulatedFrames = 0;
		m_FrameTimeSum				= 0.0;
	}
}

void DeviceManager::RunMessageLoop() {
	m_PreviousFrameTimestamp = glfwGetTime();

	while (!glfwWindowShouldClose(m_Window)) {

		if (m_callbacks.beforeFrame) m_callbacks.beforeFrame(*this);

		glfwPollEvents();
		UpdateWindowSize();

		double curTime	   = glfwGetTime();
		double elapsedTime = curTime - m_PreviousFrameTimestamp;

		if (m_windowVisible) {
			if (m_callbacks.beforeAnimate) m_callbacks.beforeAnimate(*this);
			Animate(elapsedTime);
			if (m_callbacks.afterAnimate) m_callbacks.afterAnimate(*this);
			if (m_callbacks.beforeRender) m_callbacks.beforeRender(*this);
			Render();
			if (m_callbacks.afterRender) m_callbacks.afterRender(*this);
			if (m_callbacks.beforePresent) m_callbacks.beforePresent(*this);
			Present();
			if (m_callbacks.afterPresent) m_callbacks.afterPresent(*this);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(0));

		GetDevice()->runGarbageCollection();

		UpdateAverageFrameTime(elapsedTime);
		m_PreviousFrameTimestamp = curTime;

		++m_FrameIndex;
	}

	GetDevice()->waitForIdle();
}

void DeviceManager::GetWindowDimensions(int &width, int &height) {
	width  = m_DeviceParams.backBufferWidth;
	height = m_DeviceParams.backBufferHeight;
}

const DeviceCreationParameters &DeviceManager::GetDeviceParams() {
	return m_DeviceParams;
}

void DeviceManager::UpdateWindowSize() {
	int width;
	int height;
	glfwGetWindowSize(m_Window, &width, &height);

	if (width == 0 || height == 0) {
		// window is minimized
		m_windowVisible = false;
		return;
	}

	m_windowVisible = true;

	if (int(m_DeviceParams.backBufferWidth) != width ||
		int(m_DeviceParams.backBufferHeight) != height ||
		(m_DeviceParams.vsyncEnabled != m_RequestedVSync &&
		 GetGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)) {
		// window is not minimized, and the size has changed

		BackBufferResizing();

		m_DeviceParams.backBufferWidth	= width;
		m_DeviceParams.backBufferHeight = height;
		m_DeviceParams.vsyncEnabled		= m_RequestedVSync;

		ResizeSwapChain();
		BackBufferResized();
	}

	m_DeviceParams.vsyncEnabled = m_RequestedVSync;
}

void DeviceManager::WindowPosCallback(int x, int y) {
#ifdef _WINDOWS
	if (m_DeviceParams.enablePerMonitorDPI) {
		HWND hwnd	 = glfwGetWin32Window(m_Window);
		auto monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);

		unsigned int dpiX;
		unsigned int dpiY;
		GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY);

		m_DPIScaleFactorX = dpiX / 96.f;
		m_DPIScaleFactorY = dpiY / 96.f;
	}
#endif
}

void DeviceManager::KeyboardUpdate(int key, int scancode, int action, int mods) {
	if (key == -1) {
		// filter unknown keys
		return;
	}

	for (auto it = m_vRenderPasses.crbegin(); it != m_vRenderPasses.crend(); it++) {
		bool ret = (*it)->KeyboardUpdate(key, scancode, action, mods);
		if (ret) break;
	}
}

void DeviceManager::KeyboardCharInput(unsigned int unicode, int mods) {
	for (auto it = m_vRenderPasses.crbegin(); it != m_vRenderPasses.crend(); it++) {
		bool ret = (*it)->KeyboardCharInput(unicode, mods);
		if (ret) break;
	}
}

void DeviceManager::MousePosUpdate(double xpos, double ypos) {
	xpos /= m_DPIScaleFactorX;
	ypos /= m_DPIScaleFactorY;

	for (auto it = m_vRenderPasses.crbegin(); it != m_vRenderPasses.crend(); it++) {
		bool ret = (*it)->MousePosUpdate(xpos, ypos);
		if (ret) break;
	}
}

void DeviceManager::MouseButtonUpdate(int button, int action, int mods) {
	for (auto it = m_vRenderPasses.crbegin(); it != m_vRenderPasses.crend(); it++) {
		bool ret = (*it)->MouseButtonUpdate(button, action, mods);
		if (ret) break;
	}
}

void DeviceManager::MouseScrollUpdate(double xoffset, double yoffset) {
	for (auto it = m_vRenderPasses.crbegin(); it != m_vRenderPasses.crend(); it++) {
		bool ret = (*it)->MouseScrollUpdate(xoffset, yoffset);
		if (ret) break;
	}
}

static void ApplyDeadZone(Vector2f &v, const float deadZone = 0.1f) {
	v *= std::max(length(v) - deadZone, 0.f) / (1.f - deadZone);
}

void DeviceManager::Shutdown() {
	m_SwapChainFramebuffers.clear();

	DestroyDeviceAndSwapChain();

	if (m_Window) {
		glfwDestroyWindow(m_Window);
		m_Window = nullptr;
	}

	glfwTerminate();
}

nvrhi::IFramebuffer *DeviceManager::GetCurrentFramebuffer() {
	return GetFramebuffer(GetCurrentBackBufferIndex());
}

nvrhi::IFramebuffer *DeviceManager::GetFramebuffer(uint32_t index) {
	if (index < m_SwapChainFramebuffers.size()) return m_SwapChainFramebuffers[index];

	return nullptr;
}

void DeviceManager::SetWindowTitle(const char *title) {
	assert(title);
	if (m_WindowTitle == title) return;

	glfwSetWindowTitle(m_Window, title);

	m_WindowTitle = title;
}

void DeviceManager::SetInformativeWindowTitle(const char *applicationName, const char *extraInfo) {
	std::stringstream ss;
	ss << applicationName;
	ss << " (" << nvrhi::utils::GraphicsAPIToString(GetDevice()->getGraphicsAPI());

	if (m_DeviceParams.enableDebugRuntime) {
		if (GetGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
			ss << ", VulkanValidationLayer";
		else
			ss << ", DebugRuntime";
	}

	if (m_DeviceParams.enableNvrhiValidationLayer) {
		ss << ", NvrhiValidationLayer";
	}

	ss << ")";

	double frameTime = GetAverageFrameTimeSeconds();
	if (frameTime > 0) {
		ss << " - " << std::setprecision(4) << (1.0 / frameTime) << " FPS ";
	}

	if (extraInfo) ss << extraInfo;

	SetWindowTitle(ss.str().c_str());
}

DeviceManager *DeviceManager::Create(nvrhi::GraphicsAPI api) {
	switch (api) {
		case nvrhi::GraphicsAPI::VULKAN:
			return CreateVK();
		default:
			Log(Error, "DeviceManager::Create: Unsupported Graphics API (%d)", api);
			return nullptr;
	}
}

DefaultMessageCallback &DefaultMessageCallback::GetInstance() {
	static DefaultMessageCallback Instance;
	return Instance;
}

void DefaultMessageCallback::message(nvrhi::MessageSeverity severity, const char *messageText) {
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

/***********************************Begin DeviceManagerVK***************************************/

template <typename T> static std::vector<T> setToVector(const std::unordered_set<T> &set) {
	std::vector<T> ret;
	for (const auto &s : set) {
		ret.push_back(s);
	}

	return ret;
}

bool DeviceManager_VK::createInstance() {
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
	for (const std::string &name : m_DeviceParams.requiredVulkanInstanceExtensions) {
		enabledExtensions.instance.insert(name);
	}
	for (const std::string &name : m_DeviceParams.optionalVulkanInstanceExtensions) {
		optionalExtensions.instance.insert(name);
	}

	// add layers requested by the user
	for (const std::string &name : m_DeviceParams.requiredVulkanLayers) {
		enabledExtensions.layers.insert(name);
	}
	for (const std::string &name : m_DeviceParams.optionalVulkanLayers) {
		optionalExtensions.layers.insert(name);
	}

	std::unordered_set<std::string> requiredExtensions = enabledExtensions.instance;

	// figure out which optional extensions are supported
	for (const auto &instanceExt : vk::enumerateInstanceExtensionProperties()) {
		const std::string name = instanceExt.extensionName;
		if (optionalExtensions.instance.find(name) != optionalExtensions.instance.end()) {
			enabledExtensions.instance.insert(name);
		}

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

	logMessage(m_DeviceParams.infoLogSeverity, "Enabled Vulkan instance extensions:");
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

	logMessage(m_DeviceParams.infoLogSeverity, "Enabled Vulkan layers:");
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

	const vk::Result res = vk::createInstance(&info, nullptr, &m_VulkanInstance);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan instance, error code = %s",
				   nvrhi::vulkan::resultToString(res));
		return false;
	}

	VULKAN_HPP_DEFAULT_DISPATCHER.init(m_VulkanInstance);

	return true;
}

void DeviceManager_VK::installDebugCallback() {
	auto info =
		vk::DebugReportCallbackCreateInfoEXT()
			.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning |
					  //   vk::DebugReportFlagBitsEXT::eInformation |
					  vk::DebugReportFlagBitsEXT::ePerformanceWarning)
			.setPfnCallback(vulkanDebugCallback)
			.setPUserData(this);

	vk::Result res =
		m_VulkanInstance.createDebugReportCallbackEXT(&info, nullptr, &m_DebugReportCallback);
	assert(res == vk::Result::eSuccess);
}

bool DeviceManager_VK::pickPhysicalDevice() {
	vk::Format requestedFormat = nvrhi::vulkan::convertFormat(m_DeviceParams.swapChainFormat);
	vk::Extent2D requestedExtent(m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight);

	auto devices = m_VulkanInstance.enumeratePhysicalDevices();

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
		auto surfaceCaps   = dev.getSurfaceCapabilitiesKHR(m_WindowSurface);
		auto surfaceFmts   = dev.getSurfaceFormatsKHR(m_WindowSurface);
		auto surfacePModes = dev.getSurfacePresentModesKHR(m_WindowSurface);

		if (surfaceCaps.minImageCount > m_DeviceParams.swapChainBufferCount ||
			(surfaceCaps.maxImageCount < m_DeviceParams.swapChainBufferCount &&
			 surfaceCaps.maxImageCount > 0)) {
			errorStream << std::endl << "  - cannot support the requested swap chain image count:";
			errorStream << " requested " << m_DeviceParams.swapChainBufferCount << ", available "
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
		uint32_t canPresent = dev.getSurfaceSupportKHR(m_GraphicsQueueFamily, m_WindowSurface);
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
		m_VulkanPhysicalDevice = discreteGPUs[0];
		return true;
	}

	if (!otherGPUs.empty()) {
		m_VulkanPhysicalDevice = otherGPUs[0];
		return true;
	}

	Log(Error, "%s", errorStream.str().c_str());

	return false;
}

bool DeviceManager_VK::findQueueFamilies(vk::PhysicalDevice physicalDevice) {
	auto props = physicalDevice.getQueueFamilyProperties();

	for (int i = 0; i < int(props.size()); i++) {
		const auto &queueFamily = props[i];

		if (m_GraphicsQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				m_GraphicsQueueFamily = i;
			}
		}

		if (m_ComputeQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				m_ComputeQueueFamily = i;
			}
		}

		if (m_TransferQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				(queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
				m_TransferQueueFamily = i;
			}
		}

		if (m_PresentQueueFamily == -1) {
			if (queueFamily.queueCount > 0 &&
				glfwGetPhysicalDevicePresentationSupport(m_VulkanInstance, physicalDevice, i)) {
				m_PresentQueueFamily = i;
			}
		}
	}

	if (m_GraphicsQueueFamily == -1 || m_PresentQueueFamily == -1 ||
		(m_ComputeQueueFamily == -1 && m_DeviceParams.enableComputeQueue) ||
		(m_TransferQueueFamily == -1 && m_DeviceParams.enableCopyQueue)) {
		return false;
	}

	return true;
}

bool DeviceManager_VK::createDevice() {
	// figure out which optional extensions are supported
	auto deviceExtensions = m_VulkanPhysicalDevice.enumerateDeviceExtensionProperties();
	for (const auto &ext : deviceExtensions) {
		const std::string name = ext.extensionName;
		if (optionalExtensions.device.find(name) != optionalExtensions.device.end()) {
			enabledExtensions.device.insert(name);
		}

		if (m_DeviceParams.enableRayTracingExtensions &&
			m_RayTracingExtensions.find(name) != m_RayTracingExtensions.end()) {
			enabledExtensions.device.insert(name);
		}
	}

	bool accelStructSupported	= false;
	bool bufferAddressSupported = false;
	bool rayPipelineSupported	= false;
	bool rayQuerySupported		= false;
	bool meshletsSupported		= false;
	bool vrsSupported			= false;

	logMessage(m_DeviceParams.infoLogSeverity, "Enabled Vulkan device extensions:");
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

	std::unordered_set<int> uniqueQueueFamilies = {m_GraphicsQueueFamily, m_PresentQueueFamily};

	if (m_DeviceParams.enableComputeQueue) uniqueQueueFamilies.insert(m_ComputeQueueFamily);

	if (m_DeviceParams.enableCopyQueue) uniqueQueueFamilies.insert(m_TransferQueueFamily);

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

	if (m_DeviceParams.deviceCreateInfoCallback)
		m_DeviceParams.deviceCreateInfoCallback(deviceDesc);

	const vk::Result res =
		m_VulkanPhysicalDevice.createDevice(&deviceDesc, nullptr, &m_VulkanDevice);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan physical device, error code = %s",
				   nvrhi::vulkan::resultToString(res));
		return false;
	}

	m_VulkanDevice.getQueue(m_GraphicsQueueFamily, 0, &m_GraphicsQueue);
	if (m_DeviceParams.enableComputeQueue)
		m_VulkanDevice.getQueue(m_ComputeQueueFamily, 0, &m_ComputeQueue);
	if (m_DeviceParams.enableCopyQueue)
		m_VulkanDevice.getQueue(m_TransferQueueFamily, 0, &m_TransferQueue);
	m_VulkanDevice.getQueue(m_PresentQueueFamily, 0, &m_PresentQueue);

	VULKAN_HPP_DEFAULT_DISPATCHER.init(m_VulkanDevice);

	// stash the renderer string
	auto prop		 = m_VulkanPhysicalDevice.getProperties();
	m_RendererString = std::string(prop.deviceName.data());

	Log(Info, "Created Vulkan device: %s", m_RendererString.c_str());

	return true;
}

bool DeviceManager_VK::createWindowSurface() {
	const VkResult res = glfwCreateWindowSurface(m_VulkanInstance, m_Window, nullptr,
												 (VkSurfaceKHR *) &m_WindowSurface);
	if (res != VK_SUCCESS) {
		Log(Error, "Failed to create a GLFW window surface, error code = %s",
				   nvrhi::vulkan::resultToString(res));
		return false;
	}

	return true;
}

void DeviceManager_VK::destroySwapChain() {
	if (m_VulkanDevice) {
		m_VulkanDevice.waitIdle();
	}

	if (m_SwapChain) {
		m_VulkanDevice.destroySwapchainKHR(m_SwapChain);
		m_SwapChain = nullptr;
	}

	m_SwapChainImages.clear();
}

bool DeviceManager_VK::createSwapChain() {
	destroySwapChain();

	m_SwapChainFormat = {vk::Format(nvrhi::vulkan::convertFormat(m_DeviceParams.swapChainFormat)),
						 vk::ColorSpaceKHR::eSrgbNonlinear};

	vk::Extent2D extent =
		vk::Extent2D(m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight);

	std::unordered_set<uint32_t> uniqueQueues = {uint32_t(m_GraphicsQueueFamily),
												 uint32_t(m_PresentQueueFamily)};

	std::vector<uint32_t> queues = setToVector(uniqueQueues);

	const bool enableSwapChainSharing = queues.size() > 1;

	auto desc =
		vk::SwapchainCreateInfoKHR()
			.setSurface(m_WindowSurface)
			.setMinImageCount(m_DeviceParams.swapChainBufferCount)
			.setImageFormat(m_SwapChainFormat.format)
			.setImageColorSpace(m_SwapChainFormat.colorSpace)
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
			.setPresentMode(m_DeviceParams.vsyncEnabled ? vk::PresentModeKHR::eFifo
														: vk::PresentModeKHR::eImmediate)
			.setClipped(true)
			.setOldSwapchain(nullptr);

	const vk::Result res = m_VulkanDevice.createSwapchainKHR(&desc, nullptr, &m_SwapChain);
	if (res != vk::Result::eSuccess) {
		Log(Error, "Failed to create a Vulkan swap chain, error code = %s",
				   nvrhi::vulkan::resultToString(res));
		return false;
	}

	// retrieve swap chain images
	auto images = m_VulkanDevice.getSwapchainImagesKHR(m_SwapChain);
	for (auto image : images) {
		SwapChainImage sci;
		sci.image = image;

		nvrhi::TextureDesc textureDesc;
		textureDesc.width			 = m_DeviceParams.backBufferWidth;
		textureDesc.height			 = m_DeviceParams.backBufferHeight;
		textureDesc.format			 = m_DeviceParams.swapChainFormat;
		textureDesc.debugName		 = "Swap chain image";
		textureDesc.initialState	 = nvrhi::ResourceStates::Present;
		textureDesc.keepInitialState = true;
		textureDesc.isRenderTarget	 = true;

		sci.rhiHandle = m_NvrhiDevice->createHandleForNativeTexture(
			nvrhi::ObjectTypes::VK_Image, nvrhi::Object(sci.image), textureDesc);
		m_SwapChainImages.push_back(sci);
	}

	m_SwapChainIndex = 0;

	return true;
}

bool DeviceManager_VK::CreateDeviceAndSwapChain() {
	if (m_DeviceParams.enableDebugRuntime) {
		enabledExtensions.instance.insert("VK_EXT_debug_report");
		enabledExtensions.layers.insert("VK_LAYER_KHRONOS_validation");
	}

	const vk::DynamicLoader dl;
	const PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = // NOLINT(misc-misplaced-const)
		dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

#define CHECK(a)                                                                                   \
	if (!(a)) {                                                                                    \
		return false;                                                                              \
	}

	CHECK(createInstance())

	if (m_DeviceParams.enableDebugRuntime) {
		installDebugCallback();
	}

	if (m_DeviceParams.swapChainFormat == nvrhi::Format::SRGBA8_UNORM)
		m_DeviceParams.swapChainFormat = nvrhi::Format::SBGRA8_UNORM;
	else if (m_DeviceParams.swapChainFormat == nvrhi::Format::RGBA8_UNORM)
		m_DeviceParams.swapChainFormat = nvrhi::Format::BGRA8_UNORM;

	// add device extensions requested by the user
	for (const std::string &name : m_DeviceParams.requiredVulkanDeviceExtensions) {
		enabledExtensions.device.insert(name);
	}
	for (const std::string &name : m_DeviceParams.optionalVulkanDeviceExtensions) {
		optionalExtensions.device.insert(name);
	}

	CHECK(createWindowSurface())
	CHECK(pickPhysicalDevice())
	CHECK(findQueueFamilies(m_VulkanPhysicalDevice))
	CHECK(createDevice())

	auto vecInstanceExt = stringSetToVector(enabledExtensions.instance);
	auto vecLayers		= stringSetToVector(enabledExtensions.layers);
	auto vecDeviceExt	= stringSetToVector(enabledExtensions.device);

	nvrhi::vulkan::DeviceDesc deviceDesc;
	deviceDesc.errorCB			  = &DefaultMessageCallback::GetInstance();
	deviceDesc.instance			  = m_VulkanInstance;
	deviceDesc.physicalDevice	  = m_VulkanPhysicalDevice;
	deviceDesc.device			  = m_VulkanDevice;
	deviceDesc.graphicsQueue	  = m_GraphicsQueue;
	deviceDesc.graphicsQueueIndex = m_GraphicsQueueFamily;
	if (m_DeviceParams.enableComputeQueue) {
		deviceDesc.computeQueue		 = m_ComputeQueue;
		deviceDesc.computeQueueIndex = m_ComputeQueueFamily;
	}
	if (m_DeviceParams.enableCopyQueue) {
		deviceDesc.transferQueue	  = m_TransferQueue;
		deviceDesc.transferQueueIndex = m_TransferQueueFamily;
	}
	deviceDesc.instanceExtensions	 = vecInstanceExt.data();
	deviceDesc.numInstanceExtensions = vecInstanceExt.size();
	deviceDesc.deviceExtensions		 = vecDeviceExt.data();
	deviceDesc.numDeviceExtensions	 = vecDeviceExt.size();

	m_NvrhiDevice = nvrhi::vulkan::createDevice(deviceDesc);

	if (m_DeviceParams.enableNvrhiValidationLayer) {
		m_ValidationLayer = nvrhi::validation::createValidationLayer(m_NvrhiDevice);
	}

	CHECK(createSwapChain())

	m_BarrierCommandList = m_NvrhiDevice->createCommandList();

	m_PresentSemaphore = m_VulkanDevice.createSemaphore(vk::SemaphoreCreateInfo());

#undef CHECK

	return true;
}

void DeviceManager_VK::DestroyDeviceAndSwapChain() {
	destroySwapChain();

	m_VulkanDevice.destroySemaphore(m_PresentSemaphore);
	m_PresentSemaphore = vk::Semaphore();

	m_BarrierCommandList = nullptr;

	m_NvrhiDevice	  = nullptr;
	m_ValidationLayer = nullptr;
	m_RendererString.clear();

	if (m_DebugReportCallback) {
		m_VulkanInstance.destroyDebugReportCallbackEXT(m_DebugReportCallback);
	}

	if (m_VulkanDevice) {
		m_VulkanDevice.destroy();
		m_VulkanDevice = nullptr;
	}

	if (m_WindowSurface) {
		assert(m_VulkanInstance);
		m_VulkanInstance.destroySurfaceKHR(m_WindowSurface);
		m_WindowSurface = nullptr;
	}

	if (m_VulkanInstance) {
		m_VulkanInstance.destroy();
		m_VulkanInstance = nullptr;
	}
}

void DeviceManager_VK::BeginFrame() {
	const vk::Result res =
		m_VulkanDevice.acquireNextImageKHR(m_SwapChain,
										   std::numeric_limits<uint64_t>::max(), // timeout
										   m_PresentSemaphore, vk::Fence(), &m_SwapChainIndex);

	assert(res == vk::Result::eSuccess);

	m_NvrhiDevice->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, m_PresentSemaphore, 0);
}

void DeviceManager_VK::Present() {
	m_NvrhiDevice->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, m_PresentSemaphore, 0);

	m_BarrierCommandList->open(); // umm...
	m_BarrierCommandList->close();
	m_NvrhiDevice->executeCommandList(m_BarrierCommandList);

	vk::PresentInfoKHR info = vk::PresentInfoKHR()
								  .setWaitSemaphoreCount(1)
								  .setPWaitSemaphores(&m_PresentSemaphore)
								  .setSwapchainCount(1)
								  .setPSwapchains(&m_SwapChain)
								  .setPImageIndices(&m_SwapChainIndex);

	const vk::Result res = m_PresentQueue.presentKHR(&info);
	assert(res == vk::Result::eSuccess || res == vk::Result::eErrorOutOfDateKHR);

	if (m_DeviceParams.enableDebugRuntime) {
		// according to vulkan-tutorial.com, "the validation layer implementation expects
		// the application to explicitly synchronize with the GPU"
		m_PresentQueue.waitIdle();
	} else {
#ifndef _WIN32
		if (m_DeviceParams.vsyncEnabled) {
			m_PresentQueue.waitIdle();
		}
#endif

		while (m_FramesInFlight.size() > m_DeviceParams.maxFramesInFlight) {
			auto query = m_FramesInFlight.front();
			m_FramesInFlight.pop();

			m_NvrhiDevice->waitEventQuery(query);

			m_QueryPool.push_back(query);
		}

		nvrhi::EventQueryHandle query;
		if (!m_QueryPool.empty()) {
			query = m_QueryPool.back();
			m_QueryPool.pop_back();
		} else {
			query = m_NvrhiDevice->createEventQuery();
		}

		m_NvrhiDevice->resetEventQuery(query);
		m_NvrhiDevice->setEventQuery(query, nvrhi::CommandQueue::Graphics);
		m_FramesInFlight.push(query);
	}
}

DeviceManager *DeviceManager::CreateVK() {
	return new DeviceManager_VK();
}

/***********************************End DeviceManagerVK***************************************/


KRR_NAMESPACE_END