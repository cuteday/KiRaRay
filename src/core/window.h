#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <nvrhi/utils.h>
#include <nvrhi/vulkan.h>
#include <nvrhi/validation.h>

#include <queue>
#include <unordered_set>

#include "imgui.h"
#include "input.h"

#include <nvrhi/vulkan.h>
#include "vulkan/binding.h"
#include "vulkan/helperpass.h"
#include "renderpass.h"

NAMESPACE_BEGIN(krr)

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
	nvrhi::Format swapChainFormat	= nvrhi::Format::RGBA8_UNORM;	// not srgb, since we render in linear HDR.
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

	[[nodiscard]] virtual vk::Device getNativeDevice() const { return mVulkanDevice; }
	[[nodiscard]] virtual nvrhi::vulkan::IDevice *getDevice() const {
		return dynamic_cast<nvrhi::vulkan::IDevice *>(mNvrhiDevice.Get());
	}
	[[nodiscard]] virtual nvrhi::IDevice *getValidationLayer() const { return mValidationLayer; }

	bool createWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
										const char *windowTitle);

	void addRenderPassToFront(RenderPass::SharedPtr pController);
	void addRenderPassToBack(RenderPass::SharedPtr pController);
	void removeRenderPass(RenderPass::SharedPtr pController);

	void runMessageLoop();

	// returns the size of the window in screen coordinates
	Vector2i getFrameSize() const;
	void getFrameSize(int &width, int &height) const;

	// returns the screen coordinate to pixel coordinate scale factor
	void getDPIScaleInfo(float &x, float &y) const {
		x = mDPIScaleFactorX;
		y = mDPIScaleFactorY;
	}

protected:
	bool mwindowVisible = false;

	DeviceCreationParameters mDeviceParams;
	GLFWwindow *mWindow = nullptr;
	// set to true if running on NV GPU
	bool mIsNvidia = false;
	std::list<RenderPass::SharedPtr> mRenderPasses;
	// timestamp in seconds for the previous frame
	double mPreviousFrameTimestamp = .0;
	// current DPI scale info (updated when window moves)
	float mDPIScaleFactorX = 1.f;
	float mDPIScaleFactorY = 1.f;
	bool mRequestedVSync   = false;

	double mAverageFrameTime		  = .0;
	double mAverageTimeUpdateInterval = .5;
	double mFrameTimeSum			  = .0;
	int mNumberOfAccumulatedFrames	  = 0;
	uint32_t mFrameIndex = 0;
	
	nvrhi::vulkan::DeviceHandle mNvrhiDevice;
	nvrhi::DeviceHandle mValidationLayer;
	nvrhi::CommandListHandle mCommandList;
	vkrhi::CuVkSemaphore mPresentSemaphore;
	
	std::vector<nvrhi::FramebufferHandle> mSwapChainFramebuffers;
	std::unique_ptr<CommonRenderPasses> mHelperPass;
	std::unique_ptr<BindingCache> mBindingCache;
	RenderContext::SharedPtr mRenderContext;

	void updateWindowSize();

	virtual void backBufferResizing();
	virtual void backBufferResized();

	virtual void tick(double elapsedTime);
	virtual void render();
	virtual void updateAverageFrameTime(double elapsedTime);

	// device-specific methods
	virtual bool createDeviceAndSwapChain();
	virtual void destroyDeviceAndSwapChain();
	virtual void resizeSwapChain() { if (mVulkanDevice) createSwapChain(); }
	virtual void beginFrame();
	virtual void present();

public:
	[[nodiscard]] virtual const char *getRendererString() const { return mRendererString.c_str(); }
	const DeviceCreationParameters getDeviceParams() { return mDeviceParams; }
	[[nodiscard]] double getAverageFrameTimeSeconds() const { return mAverageFrameTime; }
	[[nodiscard]] double getPreviousFrameTimestamp() const { return mPreviousFrameTimestamp; }
	void setFrameTimeUpdateInterval(double seconds) { mAverageTimeUpdateInterval = seconds; }
	[[nodiscard]] bool isVsyncEnabled() const { return mDeviceParams.vsyncEnabled; }
	virtual void setVsyncEnabled(bool enabled) {
		mRequestedVSync = enabled; /* will be processed later */
	}
	virtual void reportLiveObjects() {}

	virtual Vector2f getMouseScale() {
		Vector2i fbSize;
		getFrameSize(fbSize[0], fbSize[1]);
		return fbSize.cast<float>().cwiseInverse();
	}
	inline Vector2i getMousePos() const {
		double x, y;
		glfwGetCursorPos(mWindow, &x, &y);
		return {(int) x, (int) y};
	}

	virtual void onWindowClose() {}
	virtual void onWindowIconify(int iconified) {}
	virtual void onWindowFocus(int focused) {}
	virtual void onWindowRefresh() {}
	virtual void onWindowPosUpdate(int xpos, int ypos);
	virtual bool onMouseEvent(io::MouseEvent &mouseEvent);
	virtual bool onKeyEvent(io::KeyboardEvent &keyEvent);

	[[nodiscard]] GLFWwindow *getWindow() const { return mWindow; }
	[[nodiscard]] size_t getFrameIndex() const { return mFrameIndex; }
	[[nodiscard]] RenderContext *getRenderContext() const { return mRenderContext.get(); }

	virtual nvrhi::ITexture *getCurrentBackBuffer() const {
		return mSwapChainImages[mSwapChainIndex].rhiHandle;
	}
	virtual nvrhi::ITexture *getBackBuffer(size_t index) const {
		if (index < mSwapChainImages.size()) return mSwapChainImages[index].rhiHandle;
		return nullptr;
	}
	virtual nvrhi::ITexture *getRenderImage() const {
		return mRenderContext->getRenderTarget()->getColorTexture()->getVulkanTexture();
	}
	virtual size_t getCurrentBackBufferIndex() { return mSwapChainIndex; }
	virtual size_t getBackBufferCount() { return mSwapChainImages.size(); }

	void shutdown();
	void setWindowTitle(const char *title);
	
	virtual bool isVulkanInstanceExtensionEnabled(const char *extensionName) const {
		return enabledExtensions.instance.find(extensionName) != enabledExtensions.instance.end();
	}
	virtual bool isVulkanDeviceExtensionEnabled(const char *extensionName) const {
		return enabledExtensions.device.find(extensionName) != enabledExtensions.device.end();
	}
	virtual bool isVulkanLayerEnabled(const char *layerName) const {
		return enabledExtensions.layers.find(layerName) != enabledExtensions.layers.end();
	}
	virtual void getEnabledVulkanInstanceExtensions(std::vector<std::string> &extensions) const {
		for (const auto &ext : enabledExtensions.instance) extensions.push_back(ext);
	}
	virtual void getEnabledVulkanDeviceExtensions(std::vector<std::string> &extensions) const {
		for (const auto &ext : enabledExtensions.device) extensions.push_back(ext);
	}
	virtual void getEnabledVulkanLayers(std::vector<std::string> &layers) const {
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
	} mcallbacks;
	
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
			VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
			VK_NV_MESH_SHADER_EXTENSION_NAME,
			VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME,
		},
	};

	VulkanExtensionSet mCudaInteropExtensions = {
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

	std::unordered_set<std::string> mRayTracingExtensions = {
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
		VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME};

	std::string mRendererString;

	vk::Instance mVulkanInstance;
	vk::DebugReportCallbackEXT mDebugReportCallback;

	vk::PhysicalDevice mVulkanPhysicalDevice;
	int mGraphicsQueueFamily = -1;
	int mComputeQueueFamily  = -1;
	int mTransferQueueFamily = -1;
	int mPresentQueueFamily  = -1;

	vk::Device mVulkanDevice;
	vk::Queue mGraphicsQueue;
	vk::Queue mComputeQueue;
	vk::Queue mTransferQueue;
	vk::Queue mPresentQueue;

	vk::SurfaceKHR mWindowSurface;

	vk::SurfaceFormatKHR mSwapChainFormat;
	vk::SwapchainKHR mSwapChain;

	struct SwapChainImage {
		vk::Image image;
		nvrhi::TextureHandle rhiHandle;
	};

	std::vector<SwapChainImage> mSwapChainImages;
	uint32_t mSwapChainIndex = -1;

	std::queue<nvrhi::EventQueryHandle> mFramesInFlight;
	std::vector<nvrhi::EventQueryHandle> mQueryPool;
	std::shared_ptr<vkrhi::CuVkHandler> mCudaHandler;

	static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugReportFlagsEXT flags,
															  VkDebugReportObjectTypeEXT objType,
															  uint64_t obj, size_t location,
															  int32_t code, const char *layerPrefix,
															  const char *msg, void *userData) {
		const DeviceManager *manager = (const DeviceManager *) userData;

		if (manager) {
			const auto &ignored = manager->mDeviceParams.ignoredVulkanValidationMessageLocations;
			const auto found	= std::find(ignored.begin(), ignored.end(), location);
			if (found != ignored.end()) return VK_FALSE;
		}

		Log(Warning, "[Vulkan: location=0x%zx code=%d, layerPrefix='%s'] %s", location, code,
			layerPrefix, msg);

		return VK_FALSE;
	}

	std::string mWindowTitle;
};

NAMESPACE_END(krr)