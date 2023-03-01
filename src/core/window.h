#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <nvrhi/utils.h>
#include <nvrhi/vulkan.h>
#include <nvrhi/validation.h>

#include <queue>
#include <unordered_set>

#include "imgui.h"
#include "input.h"
#include "device/buffer.h"

#include <cuda_runtime.h>

#include <nvrhi/vulkan.h>
#include "vulkan/binding.h"
#include "vulkan/helperpass.h"
#include "renderpass.h"

KRR_NAMESPACE_BEGIN

namespace ui = ImGui;

class DeviceManager;

struct DeviceCreationParameters {
	bool startMaximized				= false;
	bool startFullscreen			= false;
	bool allowModeSwitch			= true;
	int windowPosX					= -1; // -1 means use default placement
	int windowPosY					= -1;
	uint32_t backBufferWidth		= 1280;
	uint32_t backBufferHeight		= 720;
	uint32_t refreshRate			= 0;
	uint32_t swapChainBufferCount	= 2;
	nvrhi::Format swapChainFormat	= nvrhi::Format::SRGBA8_UNORM;
	nvrhi::Format renderFormat		= nvrhi::Format::RGBA32_FLOAT;
	uint32_t swapChainSampleCount	= 1;
	uint32_t swapChainSampleQuality = 0;
	uint32_t maxFramesInFlight		= 2;
	bool enableDebugRuntime			= true;
	bool enableNvrhiValidationLayer = true;
	bool vsyncEnabled				= false;
	bool enableRayTracingExtensions = false;
	bool enableComputeQueue			= true;
	bool enableCopyQueue			= true;
	bool enablePerMonitorDPI		= false;
	bool enableCudaInterop			= true;

	Log::Level infoLogSeverity = Log::Level::Info;

	std::vector<std::string> requiredVulkanInstanceExtensions;
	std::vector<std::string> requiredVulkanDeviceExtensions;
	std::vector<std::string> requiredVulkanLayers;
	std::vector<std::string> optionalVulkanInstanceExtensions;
	std::vector<std::string> optionalVulkanDeviceExtensions;
	std::vector<std::string> optionalVulkanLayers;
	std::vector<size_t> ignoredVulkanValidationMessageLocations;
	std::function<void(vk::DeviceCreateInfo &)> deviceCreateInfoCallback;
};


class DeviceManager {
public:
	DeviceManager()			 = default;
	virtual ~DeviceManager() = default;

	[[nodiscard]] virtual vk::Device GetNativeDevice() const { return m_VulkanDevice; }

	[[nodiscard]] virtual nvrhi::vulkan::IDevice *GetDevice() const {
		return dynamic_cast<nvrhi::vulkan::IDevice*>(m_NvrhiDevice.Get());
	}

	[[nodiscard]] virtual nvrhi::IDevice* GetValidationLayer() const {
		return m_ValidationLayer;
	}

	bool CreateWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
										const char *windowTitle);

	void AddRenderPassToFront(RenderPass::SharedPtr pController);
	void AddRenderPassToBack(RenderPass::SharedPtr pController);
	void RemoveRenderPass(RenderPass::SharedPtr pController);

	void RunMessageLoop();

	// returns the size of the window in screen coordinates
	Vector2i GetWindowDimensions() const;
	void GetWindowDimensions(int &width, int &height) const;

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
	std::list<RenderPass::SharedPtr> m_RenderPasses;
	// timestamp in seconds for the previous frame
	double m_PreviousFrameTimestamp = .0;
	// current DPI scale info (updated when window moves)
	float m_DPIScaleFactorX = 1.f;
	float m_DPIScaleFactorY = 1.f;
	bool m_RequestedVSync	= false;

	double m_AverageFrameTime		  = .0;
	double m_AverageTimeUpdateInterval= .5;
	double m_FrameTimeSum			  = .0;
	int m_NumberOfAccumulatedFrames	  =  0;

	uint32_t m_FrameIndex;

	std::vector<nvrhi::FramebufferHandle> m_SwapChainFramebuffers;
	std::vector<RenderFrame::SharedPtr> m_RenderFramebuffers;

	std::unique_ptr<CommonRenderPasses> m_HelperPass;
	std::unique_ptr<BindingCache> m_BindingCache;

	void UpdateWindowSize();

	virtual void BackBufferResizing();
	virtual void BackBufferResized();

	virtual void Tick(double elapsedTime);
	virtual void Render();
	virtual void UpdateAverageFrameTime(double elapsedTime);

	// device-specific methods
	virtual bool CreateDeviceAndSwapChain();
	virtual void DestroyDeviceAndSwapChain();
	virtual void ResizeSwapChain() { if (m_VulkanDevice) createSwapChain(); }
	virtual void BeginFrame();
	virtual void Present();

public:
	[[nodiscard]] virtual const char *GetRendererString() const { return m_RendererString.c_str(); }
	const DeviceCreationParameters &GetDeviceParams() { return m_DeviceParams; }
	[[nodiscard]] double GetAverageFrameTimeSeconds() const { return m_AverageFrameTime; }
	[[nodiscard]] double GetPreviousFrameTimestamp() const { return m_PreviousFrameTimestamp; }
	void SetFrameTimeUpdateInterval(double seconds) { m_AverageTimeUpdateInterval = seconds; }
	[[nodiscard]] bool IsVsyncEnabled() const { return m_DeviceParams.vsyncEnabled; }
	virtual void SetVsyncEnabled(bool enabled) {
		m_RequestedVSync = enabled; /* will be processed later */
	}
	virtual void ReportLiveObjects() {}
	
	virtual Vector2f getMouseScale() { 
		Vector2i fbSize;
		GetWindowDimensions(fbSize[0], fbSize[1]);
		return fbSize.cast<float>().cwiseInverse(); 
	}
	inline Vector2i getMousePos() const {
		double x, y;
		glfwGetCursorPos(m_Window, &x, &y);
		return {(int) x, (int) y};
	}
	
	virtual void onWindowClose() {}
	virtual void onWindowIconify(int iconified) {}
	virtual void onWindowFocus(int focused) {}
	virtual void onWindowRefresh() {}
	virtual void onWindowPosUpdate(int xpos, int ypos);
	virtual bool onMouseEvent(io::MouseEvent &mouseEvent);
	virtual bool onKeyEvent(io::KeyboardEvent &keyEvent);

	[[nodiscard]] GLFWwindow *GetWindow() const { return m_Window; }
	[[nodiscard]] size_t GetFrameIndex() const { return m_FrameIndex; }

	virtual nvrhi::ITexture *GetCurrentBackBuffer() const {
		return m_SwapChainImages[m_SwapChainIndex].rhiHandle;
	}
	virtual nvrhi::ITexture *GetBackBuffer(size_t index) const {
		if (index < m_SwapChainImages.size()) return m_SwapChainImages[index].rhiHandle;
		return nullptr;
	}
	virtual nvrhi::ITexture *GetCurrentRenderImage() const {
		return m_RenderImages[m_SwapChainIndex];
	}
	virtual nvrhi::ITexture *GetRenderImage(size_t index) const {
		if (index < m_RenderImages.size()) return m_RenderImages[index];
		return nullptr;
	}
	virtual size_t GetCurrentBackBufferIndex() { return m_SwapChainIndex; }
	virtual size_t GetBackBufferCount() { return m_SwapChainImages.size(); }

	void Shutdown();
	void SetWindowTitle(const char *title);
	
	virtual bool IsVulkanInstanceExtensionEnabled(const char *extensionName) const {
		return enabledExtensions.instance.find(extensionName) != enabledExtensions.instance.end();
	}
	virtual bool IsVulkanDeviceExtensionEnabled(const char *extensionName) const {
		return enabledExtensions.device.find(extensionName) != enabledExtensions.device.end();
	}
	virtual bool IsVulkanLayerEnabled(const char *layerName) const {
		return enabledExtensions.layers.find(layerName) != enabledExtensions.layers.end();
	}
	virtual void GetEnabledVulkanInstanceExtensions(std::vector<std::string> &extensions) const {
		for (const auto &ext : enabledExtensions.instance) extensions.push_back(ext);
	}
	virtual void GetEnabledVulkanDeviceExtensions(std::vector<std::string> &extensions) const {
		for (const auto &ext : enabledExtensions.device) extensions.push_back(ext);
	}
	virtual void GetEnabledVulkanLayers(std::vector<std::string> &layers) const {
		for (const auto &ext : enabledExtensions.layers) layers.push_back(ext);
	}

	struct PipelineCallbacks {
		std::function<void(DeviceManager &)> beforeFrame   = nullptr;
		std::function<void(DeviceManager &)> beforeTick = nullptr;
		std::function<void(DeviceManager &)> afterTick  = nullptr;
		std::function<void(DeviceManager &)> beforeRender  = nullptr;
		std::function<void(DeviceManager &)> afterRender   = nullptr;
		std::function<void(DeviceManager &)> beforePresent = nullptr;
		std::function<void(DeviceManager &)> afterPresent  = nullptr;
	} m_callbacks;


	
protected:
	nvrhi::vulkan::DeviceHandle m_NvrhiDevice;
	nvrhi::DeviceHandle m_ValidationLayer;

	nvrhi::CommandListHandle m_CommandList;
	vkrhi::CuVkSemaphore m_PresentSemaphore;
	vkrhi::CuVkSemaphore m_GraphicsSemaphore;
	
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
		{VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
		// layers
		{},
		// device
		{
			VK_EXT_DEBUG_MARKER_EXTENSION_NAME,
			VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
			VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
			VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
			VK_NV_MESH_SHADER_EXTENSION_NAME,
			VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME,
		},
	};

	VulkanExtensionSet m_CudaInteropExtensions = {
		{VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
		 VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,},
		{},
		{
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
			VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
		}};

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
	uint32_t m_SwapChainIndex = -1;

	std::queue<nvrhi::EventQueryHandle> m_FramesInFlight;
	std::vector<nvrhi::EventQueryHandle> m_QueryPool;

	std::shared_ptr<vkrhi::CuVkHandler> m_CuVkHandler;

	static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugReportFlagsEXT flags,
															  VkDebugReportObjectTypeEXT objType,
															  uint64_t obj, size_t location,
															  int32_t code, const char *layerPrefix,
															  const char *msg, void *userData) {
		const DeviceManager *manager = (const DeviceManager *) userData;

		if (manager) {
			const auto &ignored = manager->m_DeviceParams.ignoredVulkanValidationMessageLocations;
			const auto found	= std::find(ignored.begin(), ignored.end(), location);
			if (found != ignored.end()) return VK_FALSE;
		}

		Log(Warning, "[Vulkan: location=0x%zx code=%d, layerPrefix='%s'] %s", location, code,
			layerPrefix, msg);

		return VK_FALSE;
	}

	std::string m_WindowTitle;
};

KRR_NAMESPACE_END