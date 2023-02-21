#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <nvrhi/utils.h>
#include <nvrhi/vulkan.h>
#include <nvrhi/validation.h>

#define GLFW_INCLUDE_NONE // Do not include any OpenGL headers
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif // _WIN32
#include <GLFW/glfw3native.h>

#include <list>
#include <queue>
#include <unordered_set>
#include <functional>

#include <common.h>
#include <logger.h>

#include "vulkan/cufriends.h"
#include "vulkan/binding.h"
#include "vulkan/helperpass.h"

KRR_NAMESPACE_BEGIN

class RenderFrame {
public:
	using SharedPtr = std::shared_ptr<RenderFrame>;

	RenderFrame(nvrhi::FramebufferHandle &framebuffer) : 
		m_framebuffer (framebuffer){}
		
	operator nvrhi::FramebufferHandle() const { return m_framebuffer; }
	operator nvrhi::IFramebuffer *() const { return m_framebuffer.Get(); }

	nvrhi::FramebufferHandle getFramebuffer() const { return m_framebuffer; }

	cudaSurfaceObject_t getMappedCudaSurface(std::shared_ptr<vkrhi::CudaVulkanFriend> cuFriend) {
		if (!m_cudaFrame) {
			m_cudaFrame = cuFriend->mapVulkanTextureToCudaSurface(
				m_framebuffer->getDesc().colorAttachments[0].texture, 
				cudaArrayColorAttachment);
		}
		return m_cudaFrame;
	}
	
	nvrhi::FramebufferHandle m_framebuffer;
	cudaSurfaceObject_t m_cudaFrame{};
};

struct DefaultMessageCallback : public nvrhi::IMessageCallback {
	static DefaultMessageCallback &GetInstance();

	void message(nvrhi::MessageSeverity severity, const char *messageText) override;
};

struct DeviceCreationParameters {
	bool startMaximized				= false;
	bool startFullscreen			= false;
	bool allowModeSwitch			= true;
	int windowPosX					= -1; // -1 means use default placement
	int windowPosY					= -1;
	uint32_t backBufferWidth		= 1280;
	uint32_t backBufferHeight		= 720;
	uint32_t refreshRate			= 0;
	uint32_t swapChainBufferCount	= 3;
	nvrhi::Format swapChainFormat	= nvrhi::Format::SRGBA8_UNORM;
	nvrhi::Format renderFormat		= nvrhi::Format::RGBA32_FLOAT; 
	uint32_t swapChainSampleCount	= 1;
	uint32_t swapChainSampleQuality = 0;
	uint32_t maxFramesInFlight		= 2;
	bool enableDebugRuntime			= false;
	bool enableNvrhiValidationLayer = false;
	bool vsyncEnabled				= false;
	bool enableRayTracingExtensions = false; 
	bool enableComputeQueue			= false;
	bool enableCopyQueue			= false;
	bool enableCudaInterop			= false;

	// Severity of the information log messages from the device manager.
	Log::Level infoLogSeverity = Log::Level::Info;

	// For use in the case of multiple adapters; only effective if 'adapter' is null. If this is
	// non-null, device creation will try to match the given string against an adapter name.  If the
	// specified string exists as a sub-string of the adapter name, the device and window will be
	// created on that adapter.  Case sensitive.
	std::wstring adapterNameSubstring = L"";

	// set to true to enable DPI scale factors to be computed per monitor
	// this will keep the on-screen window size in pixels constant
	//
	// if set to false, the DPI scale factors will be constant but the system
	// may scale the contents of the window based on DPI
	//
	// note that the backbuffer size is never updated automatically; if the app
	// wishes to scale up rendering based on DPI, then it must set this to true
	// and respond to DPI scale factor changes by resizing the backbuffer explicitly
	bool enablePerMonitorDPI = false;

	std::vector<std::string> requiredVulkanInstanceExtensions;
	std::vector<std::string> requiredVulkanDeviceExtensions;
	std::vector<std::string> requiredVulkanLayers;
	std::vector<std::string> optionalVulkanInstanceExtensions;
	std::vector<std::string> optionalVulkanDeviceExtensions;
	std::vector<std::string> optionalVulkanLayers;
	std::vector<size_t> ignoredVulkanValidationMessageLocations;
	std::function<void(vk::DeviceCreateInfo &)> deviceCreateInfoCallback;
};

class IRenderPass;
class DeviceManagerImpl;

class DeviceManager {
public:
	static DeviceManagerImpl *Create(nvrhi::GraphicsAPI api);

	bool CreateWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
										const char *windowTitle);

	void AddRenderPassToFront(IRenderPass *pController);
	void AddRenderPassToBack(IRenderPass *pController);
	void RemoveRenderPass(IRenderPass *pController);

	void RunMessageLoop();

	// returns the size of the window in screen coordinates
	void GetWindowDimensions(int &width, int &height);
	// returns the screen coordinate to pixel coordinate scale factor
	void GetDPIScaleInfo(float &x, float &y) const {
		x = m_DPIScaleFactorX;
		y = m_DPIScaleFactorY;
	}

protected:
	bool m_windowVisible = false;

	DeviceCreationParameters m_DeviceParams;
	GLFWwindow *m_Window = nullptr;
	// set to true if running on NV GPU
	bool m_IsNvidia = false;
	std::list<IRenderPass *> m_vRenderPasses;
	// timestamp in seconds for the previous frame
	double m_PreviousFrameTimestamp = 0.0;
	// current DPI scale info (updated when window moves)
	float m_DPIScaleFactorX = 1.f;
	float m_DPIScaleFactorY = 1.f;
	bool m_RequestedVSync	= false;

	double m_AverageFrameTime		   = 0.0;
	double m_AverageTimeUpdateInterval = 0.5;
	double m_FrameTimeSum			   = 0.0;
	int m_NumberOfAccumulatedFrames	   = 0;

	uint32_t m_FrameIndex = 0;

	std::vector<nvrhi::FramebufferHandle> m_SwapChainFramebuffers;
	std::vector<RenderFrame::SharedPtr> m_RenderFramebuffers;
	
	std::unique_ptr<CommonRenderPasses> m_HelperPass;
	std::unique_ptr<BindingCache> m_BindingCache;

	DeviceManager() = default;

	void UpdateWindowSize();

	void BackBufferResizing();
	void BackBufferResized();

	void Animate(double elapsedTime);
	void Render();
	void UpdateAverageFrameTime(double elapsedTime);

	// device-specific methods
	virtual bool CreateDeviceAndSwapChain()	 = 0;
	virtual void DestroyDeviceAndSwapChain() = 0;
	virtual void ResizeSwapChain()			 = 0;
	virtual void BeginFrame()				 = 0;
	virtual void Present()					 = 0;

public:
	[[nodiscard]] virtual nvrhi::IDevice *GetDevice(bool withValidationLayer = true) const = 0;
	[[nodiscard]] virtual const char *GetRendererString() const		= 0;
	[[nodiscard]] virtual nvrhi::GraphicsAPI GetGraphicsAPI() const = 0;

	const DeviceCreationParameters &GetDeviceParams();
	[[nodiscard]] double GetAverageFrameTimeSeconds() const {
		return m_AverageFrameTime;
	}
	[[nodiscard]] double GetPreviousFrameTimestamp() const {
		return m_PreviousFrameTimestamp;
	}
	void SetFrameTimeUpdateInterval(double seconds) {
		m_AverageTimeUpdateInterval = seconds;
	}
	[[nodiscard]] bool IsVsyncEnabled() const {
		return m_DeviceParams.vsyncEnabled;
	}
	virtual void SetVsyncEnabled(bool enabled) {
		m_RequestedVSync = enabled; /* will be processed later */
	}
	virtual void ReportLiveObjects() {}

	// these are public in order to be called from the GLFW callback functions
	void WindowCloseCallback() {}
	void WindowIconifyCallback(int iconified) {}
	void WindowFocusCallback(int focused) {}
	void WindowRefreshCallback() {}
	void WindowPosCallback(int xpos, int ypos);

	void KeyboardUpdate(int key, int scancode, int action, int mods);
	void KeyboardCharInput(unsigned int unicode, int mods);
	void MousePosUpdate(double xpos, double ypos);
	void MouseButtonUpdate(int button, int action, int mods);
	void MouseScrollUpdate(double xoffset, double yoffset);

	[[nodiscard]] GLFWwindow *GetWindow() const {
		return m_Window;
	}
	[[nodiscard]] uint32_t GetFrameIndex() const {
		return m_FrameIndex;
	}

	virtual nvrhi::ITexture *GetCurrentBackBuffer()		   = 0;
	virtual nvrhi::ITexture *GetBackBuffer(uint32_t index) = 0;
	virtual nvrhi::ITexture *GetRenderImage(uint32_t index)= 0;
	virtual uint32_t GetCurrentBackBufferIndex()		   = 0;
	virtual uint32_t GetBackBufferCount()				   = 0;
	nvrhi::IFramebuffer *GetCurrentFramebuffer();
	nvrhi::IFramebuffer *GetFramebuffer(uint32_t index);

	void Shutdown();
	virtual ~DeviceManager() = default;

	void SetWindowTitle(const char *title);
	void SetInformativeWindowTitle(const char *applicationName, const char *extraInfo = nullptr);

	virtual bool IsVulkanInstanceExtensionEnabled(const char *extensionName) const { return false; }
	virtual bool IsVulkanDeviceExtensionEnabled(const char *extensionName) const { return false; }
	virtual bool IsVulkanLayerEnabled(const char *layerName) const { return false; }
	virtual void GetEnabledVulkanInstanceExtensions(std::vector<std::string> &extensions) const {}
	virtual void GetEnabledVulkanDeviceExtensions(std::vector<std::string> &extensions) const {}
	virtual void GetEnabledVulkanLayers(std::vector<std::string> &layers) const {}

	struct PipelineCallbacks {
		std::function<void(DeviceManager &)> beforeFrame   = nullptr;
		std::function<void(DeviceManager &)> beforeAnimate = nullptr;
		std::function<void(DeviceManager &)> afterAnimate  = nullptr;
		std::function<void(DeviceManager &)> beforeRender  = nullptr;
		std::function<void(DeviceManager &)> afterRender   = nullptr;
		std::function<void(DeviceManager &)> beforePresent = nullptr;
		std::function<void(DeviceManager &)> afterPresent  = nullptr;
	} m_callbacks;

private:
	static DeviceManagerImpl *CreateImpl();

	std::string m_WindowTitle;
};

class DeviceManagerImpl : public DeviceManager {
public:
	[[nodiscard]] vk::Device GetNativeDevice() const { return m_VulkanDevice; }

	[[nodiscard]] nvrhi::IDevice *GetDevice(bool withValidationLayer = true) const override {
		if (withValidationLayer && m_ValidationLayer) return m_ValidationLayer;

		return m_NvrhiDevice;
	}

	[[nodiscard]] nvrhi::GraphicsAPI GetGraphicsAPI() const override {
		return nvrhi::GraphicsAPI::VULKAN;
	}

protected:
	bool CreateDeviceAndSwapChain() override;
	void DestroyDeviceAndSwapChain() override;

	void ResizeSwapChain() override {
		if (m_VulkanDevice) {
			// destroySwapChain();
			createSwapChain();
		}
	}

	nvrhi::ITexture *GetCurrentBackBuffer() override {
		return m_SwapChainImages[m_SwapChainIndex].rhiHandle;
	}
	nvrhi::ITexture *GetBackBuffer(uint32_t index) override {
		if (index < m_SwapChainImages.size()) return m_SwapChainImages[index].rhiHandle;
		return nullptr;
	}
	nvrhi::ITexture *GetRenderImage(uint32_t index) override {
		if (index < m_RenderImages.size()) return m_RenderImages[index];
		return nullptr;
	}
	uint32_t GetCurrentBackBufferIndex() override {
		return m_SwapChainIndex;
	}
	uint32_t GetBackBufferCount() override {
		return uint32_t(m_SwapChainImages.size());
	}

	void BeginFrame() override;
	void Present() override;

	const char *GetRendererString() const override {
		return m_RendererString.c_str();
	}

	bool IsVulkanInstanceExtensionEnabled(const char *extensionName) const override {
		return enabledExtensions.instance.find(extensionName) != enabledExtensions.instance.end();
	}

	bool IsVulkanDeviceExtensionEnabled(const char *extensionName) const override {
		return enabledExtensions.device.find(extensionName) != enabledExtensions.device.end();
	}

	bool IsVulkanLayerEnabled(const char *layerName) const override {
		return enabledExtensions.layers.find(layerName) != enabledExtensions.layers.end();
	}

	void GetEnabledVulkanInstanceExtensions(std::vector<std::string> &extensions) const override {
		for (const auto &ext : enabledExtensions.instance) extensions.push_back(ext);
	}

	void GetEnabledVulkanDeviceExtensions(std::vector<std::string> &extensions) const override {
		for (const auto &ext : enabledExtensions.device) extensions.push_back(ext);
	}

	void GetEnabledVulkanLayers(std::vector<std::string> &layers) const override {
		for (const auto &ext : enabledExtensions.layers) layers.push_back(ext);
	}

private:
	bool createInstance();
	bool createWindowSurface();
	void installDebugCallback();
	bool pickPhysicalDevice();
	bool findQueueFamilies(vk::PhysicalDevice physicalDevice);
	bool createDevice();
	bool createSwapChain();
	void destroySwapChain();

	struct VulkanExtensionSet {
		std::unordered_set<std::string> instance;
		std::unordered_set<std::string> layers;
		std::unordered_set<std::string> device;
	};

	// minimal set of required extensions
	VulkanExtensionSet enabledExtensions = {
		// instance
		{VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME},
		// layers
		{},
		// device
		{VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE1_EXTENSION_NAME},
	};

	// optional extensions
	VulkanExtensionSet optionalExtensions = {
		// instance
		{VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME, 
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
		// layers
		{},
		// device
		{VK_EXT_DEBUG_MARKER_EXTENSION_NAME, 
		VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,	
		VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, 
		VK_NV_MESH_SHADER_EXTENSION_NAME,
		VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, 
		VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
		},
	};

	VulkanExtensionSet m_CudaInteropExtensions = {
		{},
		//{VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
		//VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,},
		{},
		{ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
			VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
		}
	};

	std::unordered_set<std::string> m_RayTracingExtensions = {
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
		VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME};

	std::string m_RendererString;

	vk::Instance m_VulkanInstance;
	vk::DebugReportCallbackEXT m_DebugReportCallback;

	vk::PhysicalDevice m_VulkanPhysicalDevice;
	int m_GraphicsQueueFamily = -1;
	int m_ComputeQueueFamily  = -1;
	int m_TransferQueueFamily = -1;
	int m_PresentQueueFamily  = -1;

	vk::Device m_VulkanDevice;
	vk::Queue m_GraphicsQueue;
	vk::Queue m_ComputeQueue;
	vk::Queue m_TransferQueue;
	vk::Queue m_PresentQueue;

	vk::SurfaceKHR m_WindowSurface;

	vk::SurfaceFormatKHR m_SwapChainFormat;
	vk::SwapchainKHR m_SwapChain;

	struct SwapChainImage {
		vk::Image image;
		nvrhi::TextureHandle rhiHandle;
	};

	std::vector<SwapChainImage> m_SwapChainImages;
	std::vector<nvrhi::TextureHandle> m_RenderImages;
	uint32_t m_SwapChainIndex = uint32_t(-1);

	nvrhi::vulkan::DeviceHandle m_NvrhiDevice;
	nvrhi::DeviceHandle m_ValidationLayer;

	nvrhi::CommandListHandle m_CommandList;
	vk::Semaphore m_PresentSemaphore;

	std::queue<nvrhi::EventQueryHandle> m_FramesInFlight;
	std::vector<nvrhi::EventQueryHandle> m_QueryPool;

	std::shared_ptr<vkrhi::CudaVulkanFriend> m_CUFriend;

private:
	static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugReportFlagsEXT flags,
															  VkDebugReportObjectTypeEXT objType,
															  uint64_t obj, size_t location,
															  int32_t code, const char *layerPrefix,
															  const char *msg, void *userData) {
		const DeviceManagerImpl *manager = (const DeviceManagerImpl *) userData;

		if (manager) {
			const auto &ignored = manager->m_DeviceParams.ignoredVulkanValidationMessageLocations;
			const auto found	= std::find(ignored.begin(), ignored.end(), location);
			if (found != ignored.end()) return VK_FALSE;
		}

		Log(Warning, "[Vulkan: location=0x%zx code=%d, layerPrefix='%s'] %s", location, code,
			layerPrefix, msg);

		return VK_FALSE;
	}
};

static std::vector<const char *> stringSetToVector(const std::unordered_set<std::string> &set) {
	std::vector<const char *> ret;
	for (const auto &s : set) {
		ret.push_back(s.c_str());
	}
	return ret;
}

class IRenderPass {
private:
	DeviceManagerImpl *m_DeviceManager;

public:
	explicit IRenderPass(DeviceManagerImpl *deviceManager) : m_DeviceManager(deviceManager) {}

	virtual ~IRenderPass() = default;

	virtual void Render(RenderFrame::SharedPtr framebuffer) {}
	virtual void Animate(float fElapsedTimeSeconds) {}
	virtual void BackBufferResizing() {}
	virtual void BackBufferResized(const uint32_t width, const uint32_t height,
								   const uint32_t sampleCount) {}

	// all of these pass in GLFW constants as arguments
	// see http://www.glfw.org/docs/latest/input.html
	// return value is true if the event was consumed by this render pass, false if it should be
	// passed on
	virtual bool KeyboardUpdate(int key, int scancode, int action, int mods) { return false; }
	virtual bool KeyboardCharInput(unsigned int unicode, int mods) { return false; }
	virtual bool MousePosUpdate(double xpos, double ypos) { return false; }
	virtual bool MouseScrollUpdate(double xoffset, double yoffset) { return false; }
	virtual bool MouseButtonUpdate(int button, int action, int mods) { return false; }

	[[nodiscard]] vk::Device GetNativeDevice() const { return m_DeviceManager->GetNativeDevice(); }

	[[nodiscard]] DeviceManager *GetDeviceManager() const {
		return m_DeviceManager;
	}
	[[nodiscard]] nvrhi::vulkan::IDevice *GetVkDevice() const {
		return dynamic_cast<nvrhi::vulkan::IDevice *>(m_DeviceManager->GetDevice(false));
	}
	[[nodiscard]] nvrhi::IDevice *GetDevice(bool withValidationLayer = true) const {
		return m_DeviceManager->GetDevice(withValidationLayer);
	}
	[[nodiscard]] uint32_t GetFrameIndex() const {
		return m_DeviceManager->GetFrameIndex();
	}
};

KRR_NAMESPACE_END