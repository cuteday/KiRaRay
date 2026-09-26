#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <nvrhi/vulkan.h>
#include <cuda_runtime.h>
#include <cstring>
#include <limits>
#include <set>

#include "graphics/device_backend.h"
#include "graphics/interop.h"
#include "logger.h"
#include "util/check.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

NAMESPACE_BEGIN(krr)

namespace {

void checkVulkan(VkResult result, const char *operation) {
	if (result != VK_SUCCESS)
		throw std::runtime_error(std::string(operation) + ": " + nvrhi::vulkan::resultToString(result));
}

class VulkanBackend : public GraphicsBackend {
public:
	~VulkanBackend() override {
		if (mDevice) vkDeviceWaitIdle(mDevice);
		destroySwapChain();
		mNvrhiDevice = nullptr;
		if (mDevice) vkDestroyDevice(mDevice, nullptr);
		if (mSurface) vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
		if (mDebugMessenger) {
			auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(mInstance, "vkDestroyDebugUtilsMessengerEXT"));
			if (destroy) destroy(mInstance, mDebugMessenger, nullptr);
		}
		if (mInstance) vkDestroyInstance(mInstance, nullptr);
	}

	void initialize(DeviceCreationParameters &params, GLFWwindow *window,
					nvrhi::IMessageCallback *callback) override;
	nvrhi::IDevice *getDevice() const override { return mNvrhiDevice; }
	const std::string &getRendererString() const override { return mRendererString; }
	std::unique_ptr<GraphicsInterop> createInterop(nvrhi::IDevice *device) override {
		return createVulkanInterop(device, mGraphicsQueueFamily);
	}
	void resizeSwapChain(const DeviceCreationParameters &params) override;
	bool beginFrame() override;
	void present() override;
	nvrhi::ITexture *getBackBuffer(size_t index) const override {
		return index < mImages.size() ? mImages[index].Get() : nullptr;
	}
	size_t getBackBufferCount() const override { return mImages.size(); }
	size_t getCurrentBackBufferIndex() const override { return mImageIndex; }

private:
	void createInstance(bool debug);
	void createDevice(nvrhi::IMessageCallback *callback);
	void destroySwapChain();
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
		const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
		logMessage(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? Log::Level::Error :
			Log::Level::Warning, std::string("Vulkan: ") + data->pMessage);
		return VK_FALSE;
	}

	DeviceCreationParameters mParams;
	GLFWwindow *mWindow = nullptr;
	VkInstance mInstance = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
	VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
	VkDevice mDevice = VK_NULL_HANDLE;
	VkQueue mGraphicsQueue = VK_NULL_HANDLE;
	uint32_t mGraphicsQueueFamily = 0;
	VkSurfaceKHR mSurface = VK_NULL_HANDLE;
	VkSwapchainKHR mSwapChain = VK_NULL_HANDLE;
	std::vector<nvrhi::TextureHandle> mImages;
	std::vector<VkSemaphore> mAcquireSemaphores;
	std::vector<VkSemaphore> mPresentSemaphores;
	std::vector<nvrhi::EventQueryHandle> mFrameQueries;
	std::vector<bool> mFrameSubmitted;
	size_t mFrameIndex = 0;
	uint32_t mImageIndex = 0;
	nvrhi::vulkan::DeviceHandle mNvrhiDevice;
	std::string mRendererString;
	std::vector<const char *> mInstanceExtensions;
};

void VulkanBackend::createInstance(bool debug) {
	if (mWindow) {
		if (!glfwVulkanSupported()) throw std::runtime_error("GLFW cannot load Vulkan.");
		uint32_t count = 0;
		const char **extensions = glfwGetRequiredInstanceExtensions(&count);
		if (!extensions) throw std::runtime_error("GLFW cannot find Vulkan surface extensions.");
		mInstanceExtensions.assign(extensions, extensions + count);
	}

	std::vector<const char *> layers;
	if (debug) {
		uint32_t count = 0;
		checkVulkan(vkEnumerateInstanceLayerProperties(&count, nullptr), "Enumerate Vulkan layers");
		std::vector<VkLayerProperties> properties(count);
		checkVulkan(vkEnumerateInstanceLayerProperties(&count, properties.data()), "Enumerate Vulkan layers");
		for (const auto &layer : properties)
			if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
				layers.push_back("VK_LAYER_KHRONOS_validation");
		if (layers.empty()) Log(Warning, "Vulkan validation layer is unavailable.");
		else mInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	application.pApplicationName = "KiRaRay";
	application.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	info.pApplicationInfo = &application;
	info.enabledExtensionCount = uint32_t(mInstanceExtensions.size());
	info.ppEnabledExtensionNames = mInstanceExtensions.data();
	info.enabledLayerCount = uint32_t(layers.size());
	info.ppEnabledLayerNames = layers.data();
	checkVulkan(vkCreateInstance(&info, nullptr, &mInstance), "Create Vulkan instance");

	if (!layers.empty()) {
		VkDebugUtilsMessengerCreateInfoEXT messenger{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		messenger.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		messenger.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		messenger.pfnUserCallback = debugCallback;
		auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(mInstance, "vkCreateDebugUtilsMessengerEXT"));
		checkVulkan(create(mInstance, &messenger, nullptr, &mDebugMessenger), "Create Vulkan validation callback");
	}
}

void VulkanBackend::createDevice(nvrhi::IMessageCallback *callback) {
	int cudaDevice = 0;
	cudaDeviceProp cudaProperties{};
	CUDA_CHECK(cudaGetDevice(&cudaDevice));
	CUDA_CHECK(cudaGetDeviceProperties(&cudaProperties, cudaDevice));
	uint32_t count = 0;
	checkVulkan(vkEnumeratePhysicalDevices(mInstance, &count, nullptr), "Enumerate Vulkan devices");
	std::vector<VkPhysicalDevice> devices(count);
	checkVulkan(vkEnumeratePhysicalDevices(mInstance, &count, devices.data()), "Enumerate Vulkan devices");
	for (VkPhysicalDevice device : devices) {
		VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
		VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
		properties.pNext = &id;
		vkGetPhysicalDeviceProperties2(device, &properties);
		if (std::memcmp(id.deviceUUID, &cudaProperties.uuid, VK_UUID_SIZE) != 0) continue;
		if (properties.properties.apiVersion < VK_API_VERSION_1_3)
			throw std::runtime_error("The CUDA device must support Vulkan 1.3.");
		mPhysicalDevice = device;
		mRendererString = properties.properties.deviceName;
		break;
	}
	if (!mPhysicalDevice) throw std::runtime_error("No Vulkan adapter matches the active CUDA device.");

	std::vector<const char *> extensions{
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
#else
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
#endif
	};
	if (mSurface) extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	checkVulkan(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &count, nullptr), "Enumerate Vulkan extensions");
	std::vector<VkExtensionProperties> available(count);
	checkVulkan(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &count, available.data()), "Enumerate Vulkan extensions");
	for (const char *extension : extensions) {
		const bool found = std::any_of(available.begin(), available.end(), [extension](const auto &entry) {
			return std::strcmp(entry.extensionName, extension) == 0;
		});
		if (!found) throw std::runtime_error(std::string("Missing Vulkan extension: ") + extension);
	}

	VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	supported.pNext = &features12;
	features12.pNext = &features13;
	vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &supported);
	if (!features13.synchronization2 || !features13.dynamicRendering || !features12.timelineSemaphore ||
		!features12.runtimeDescriptorArray || !features12.descriptorBindingPartiallyBound ||
		!features12.descriptorBindingUpdateUnusedWhilePending)
		throw std::runtime_error("The Vulkan device lacks required synchronization or descriptor features.");
	// Enable only features used by NVRHI and the shared shaders.
	VkPhysicalDeviceVulkan13Features enabled13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	enabled13.synchronization2 = VK_TRUE;
	enabled13.dynamicRendering = VK_TRUE;
	VkPhysicalDeviceVulkan12Features enabled12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	enabled12.pNext = &enabled13;
	enabled12.timelineSemaphore = VK_TRUE;
	enabled12.descriptorIndexing = features12.descriptorIndexing;
	enabled12.runtimeDescriptorArray = VK_TRUE;
	enabled12.descriptorBindingPartiallyBound = VK_TRUE;
	enabled12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
	enabled12.descriptorBindingVariableDescriptorCount = features12.descriptorBindingVariableDescriptorCount;
	enabled12.shaderSampledImageArrayNonUniformIndexing = features12.shaderSampledImageArrayNonUniformIndexing;
	enabled12.shaderStorageBufferArrayNonUniformIndexing = features12.shaderStorageBufferArrayNonUniformIndexing;
	enabled12.descriptorBindingSampledImageUpdateAfterBind = features12.descriptorBindingSampledImageUpdateAfterBind;
	enabled12.descriptorBindingStorageBufferUpdateAfterBind = features12.descriptorBindingStorageBufferUpdateAfterBind;
	enabled12.descriptorBindingStorageImageUpdateAfterBind = features12.descriptorBindingStorageImageUpdateAfterBind;
	enabled12.descriptorBindingStorageTexelBufferUpdateAfterBind = features12.descriptorBindingStorageTexelBufferUpdateAfterBind;
	enabled12.descriptorBindingUniformTexelBufferUpdateAfterBind = features12.descriptorBindingUniformTexelBufferUpdateAfterBind;
	enabled12.bufferDeviceAddress = features12.bufferDeviceAddress;
	VkPhysicalDeviceFeatures enabled{};
	enabled.samplerAnisotropy = supported.features.samplerAnisotropy;
	enabled.textureCompressionBC = supported.features.textureCompressionBC;
	enabled.shaderImageGatherExtended = supported.features.shaderImageGatherExtended;
	enabled.imageCubeArray = supported.features.imageCubeArray;
	enabled.geometryShader = supported.features.geometryShader;
	enabled.tessellationShader = supported.features.tessellationShader;
	enabled.dualSrcBlend = supported.features.dualSrcBlend;
	enabled.fragmentStoresAndAtomics = supported.features.fragmentStoresAndAtomics;

	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &count, nullptr);
	std::vector<VkQueueFamilyProperties> families(count);
	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &count, families.data());
	uint32_t graphics = UINT32_MAX, compute = UINT32_MAX, transfer = UINT32_MAX;
	for (uint32_t index = 0; index < count; ++index) {
		const auto flags = families[index].queueFlags;
		if (!families[index].queueCount) continue;
		VkBool32 present = VK_TRUE;
		if (mSurface) checkVulkan(vkGetPhysicalDeviceSurfaceSupportKHR(mPhysicalDevice, index, mSurface, &present), "Check Vulkan presentation support");
		if (graphics == UINT32_MAX && (flags & VK_QUEUE_GRAPHICS_BIT) && present) graphics = index;
		if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT)) compute = index;
		if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) transfer = index;
	}
	if (graphics == UINT32_MAX) throw std::runtime_error("No Vulkan graphics queue supports this window.");
	if (compute == UINT32_MAX) compute = graphics;
	if (transfer == UINT32_MAX) transfer = graphics;
	mGraphicsQueueFamily = graphics;
	std::set<uint32_t> uniqueFamilies{graphics};
	if (mParams.enableComputeQueue) uniqueFamilies.insert(compute);
	if (mParams.enableCopyQueue) uniqueFamilies.insert(transfer);
	const float priority = 1.f;
	std::vector<VkDeviceQueueCreateInfo> queues;
	for (uint32_t family : uniqueFamilies) {
		VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		queue.queueFamilyIndex = family;
		queue.queueCount = 1;
		queue.pQueuePriorities = &priority;
		queues.push_back(queue);
	}
	VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	info.pNext = &enabled12;
	info.pEnabledFeatures = &enabled;
	info.queueCreateInfoCount = uint32_t(queues.size());
	info.pQueueCreateInfos = queues.data();
	info.enabledExtensionCount = uint32_t(extensions.size());
	info.ppEnabledExtensionNames = extensions.data();
	checkVulkan(vkCreateDevice(mPhysicalDevice, &info, nullptr, &mDevice), "Create Vulkan device");
	vkGetDeviceQueue(mDevice, graphics, 0, &mGraphicsQueue);

	nvrhi::vulkan::DeviceDesc desc{};
	desc.errorCB = callback;
	desc.instance = mInstance;
	desc.physicalDevice = mPhysicalDevice;
	desc.device = mDevice;
	desc.graphicsQueue = mGraphicsQueue;
	desc.graphicsQueueIndex = int(graphics);
	if (mParams.enableComputeQueue) {
		vkGetDeviceQueue(mDevice, compute, 0, &desc.computeQueue);
		desc.computeQueueIndex = int(compute);
	}
	if (mParams.enableCopyQueue) {
		vkGetDeviceQueue(mDevice, transfer, 0, &desc.transferQueue);
		desc.transferQueueIndex = int(transfer);
	}
	desc.instanceExtensions = mInstanceExtensions.data();
	desc.numInstanceExtensions = mInstanceExtensions.size();
	desc.deviceExtensions = extensions.data();
	desc.numDeviceExtensions = extensions.size();
	desc.bufferDeviceAddressSupported = enabled12.bufferDeviceAddress;
	VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance, vkGetInstanceProcAddr, mDevice);
	mNvrhiDevice = nvrhi::vulkan::createDevice(desc);
	if (!mNvrhiDevice) throw std::runtime_error("NVRHI could not create its Vulkan device.");
	Log(Success, "Created Vulkan device: %s", mRendererString.c_str());
}

void VulkanBackend::initialize(DeviceCreationParameters &params, GLFWwindow *window,
							   nvrhi::IMessageCallback *callback) {
	mWindow = window;
	mParams = params;
	createInstance(params.enableDebugRuntime);
	if (window) checkVulkan(glfwCreateWindowSurface(mInstance, window, nullptr, &mSurface), "Create Vulkan window surface");
	createDevice(callback);
	if (window) {
		if (params.swapChainFormat == nvrhi::Format::RGBA8_UNORM) params.swapChainFormat = nvrhi::Format::BGRA8_UNORM;
		else if (params.swapChainFormat == nvrhi::Format::SRGBA8_UNORM) params.swapChainFormat = nvrhi::Format::SBGRA8_UNORM;
		resizeSwapChain(params);
	}
}

void VulkanBackend::destroySwapChain() {
	mFrameQueries.clear();
	mFrameSubmitted.clear();
	mImages.clear();
	for (VkSemaphore semaphore : mAcquireSemaphores) vkDestroySemaphore(mDevice, semaphore, nullptr);
	for (VkSemaphore semaphore : mPresentSemaphores) vkDestroySemaphore(mDevice, semaphore, nullptr);
	mAcquireSemaphores.clear();
	mPresentSemaphores.clear();
	if (mSwapChain) vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
	mSwapChain = VK_NULL_HANDLE;
}

void VulkanBackend::resizeSwapChain(const DeviceCreationParameters &params) {
	if (!mSurface) return;
	checkVulkan(vkDeviceWaitIdle(mDevice), "Wait for Vulkan resize");
	mNvrhiDevice->runGarbageCollection();
	destroySwapChain();
	mParams = params;
	VkSurfaceCapabilitiesKHR capabilities{};
	checkVulkan(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &capabilities), "Query Vulkan surface capabilities");
	VkExtent2D extent{params.backBufferWidth, params.backBufferHeight};
	if (capabilities.currentExtent.width != UINT32_MAX) extent = capabilities.currentExtent;
	else {
		extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	}
	uint32_t count = 0;
	checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &count, nullptr), "Query Vulkan surface formats");
	std::vector<VkSurfaceFormatKHR> formats(count);
	checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &count, formats.data()), "Query Vulkan surface formats");
	VkSurfaceFormatKHR format{nvrhi::vulkan::convertFormat(params.swapChainFormat), VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
	if (std::none_of(formats.begin(), formats.end(), [&format](const auto &entry) {
		return entry.format == format.format && entry.colorSpace == format.colorSpace;
	})) throw std::runtime_error("The Vulkan surface does not support the requested swapchain format.");
	checkVulkan(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &count, nullptr), "Query Vulkan present modes");
	std::vector<VkPresentModeKHR> modes(count);
	checkVulkan(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &count, modes.data()), "Query Vulkan present modes");
	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if (!params.vsyncEnabled) {
		if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end()) presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
	}
	uint32_t imageCount = std::max(params.swapChainBufferCount, capabilities.minImageCount);
	if (capabilities.maxImageCount) imageCount = std::min(imageCount, capabilities.maxImageCount);
	VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
	info.surface = mSurface;
	info.minImageCount = imageCount;
	info.imageFormat = format.format;
	info.imageColorSpace = format.colorSpace;
	info.imageExtent = extent;
	info.imageArrayLayers = 1;
	info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.preTransform = capabilities.currentTransform;
	info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	info.presentMode = presentMode;
	info.clipped = VK_TRUE;
	checkVulkan(vkCreateSwapchainKHR(mDevice, &info, nullptr, &mSwapChain), "Create Vulkan swapchain");
	checkVulkan(vkGetSwapchainImagesKHR(mDevice, mSwapChain, &count, nullptr), "Query Vulkan swapchain images");
	std::vector<VkImage> images(count);
	checkVulkan(vkGetSwapchainImagesKHR(mDevice, mSwapChain, &count, images.data()), "Query Vulkan swapchain images");
	for (VkImage image : images) {
		nvrhi::TextureDesc texture;
		texture.width = extent.width;
		texture.height = extent.height;
		texture.format = params.swapChainFormat;
		texture.isRenderTarget = true;
		texture.initialState = nvrhi::ResourceStates::Present;
		texture.keepInitialState = true;
		texture.debugName = "Swapchain image";
		mImages.push_back(mNvrhiDevice->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, image, texture));
	}
	VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	mPresentSemaphores.resize(images.size());
	for (auto &semaphore : mPresentSemaphores)
		checkVulkan(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &semaphore), "Create Vulkan presentation semaphore");
	const uint32_t frames = std::max(1u, params.maxFramesInFlight);
	mAcquireSemaphores.resize(frames);
	mFrameSubmitted.resize(frames, false);
	for (auto &semaphore : mAcquireSemaphores) {
		checkVulkan(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &semaphore), "Create Vulkan acquisition semaphore");
		mFrameQueries.push_back(mNvrhiDevice->createEventQuery());
	}
	mFrameIndex = mImageIndex = 0;
}

bool VulkanBackend::beginFrame() {
	if (!mSwapChain) return true;
	if (mFrameSubmitted[mFrameIndex]) mNvrhiDevice->waitEventQuery(mFrameQueries[mFrameIndex]);
	VkResult result = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX,
		mAcquireSemaphores[mFrameIndex], VK_NULL_HANDLE, &mImageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) return false;
	if (result != VK_SUBOPTIMAL_KHR) checkVulkan(result, "Acquire Vulkan swapchain image");
	mNvrhiDevice->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, mAcquireSemaphores[mFrameIndex], 0);
	return true;
}

void VulkanBackend::present() {
	if (!mSwapChain) return;
	mNvrhiDevice->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, mPresentSemaphores[mImageIndex], 0);
	mNvrhiDevice->executeCommandLists(nullptr, 0);
	mNvrhiDevice->resetEventQuery(mFrameQueries[mFrameIndex]);
	mNvrhiDevice->setEventQuery(mFrameQueries[mFrameIndex], nvrhi::CommandQueue::Graphics);
	mFrameSubmitted[mFrameIndex] = true;
	VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
	info.waitSemaphoreCount = 1;
	info.pWaitSemaphores = &mPresentSemaphores[mImageIndex];
	info.swapchainCount = 1;
	info.pSwapchains = &mSwapChain;
	info.pImageIndices = &mImageIndex;
	const VkResult result = vkQueuePresentKHR(mGraphicsQueue, &info);
	if (result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR)
		checkVulkan(result, "Present Vulkan swapchain image");
	mFrameIndex = (mFrameIndex + 1) % mAcquireSemaphores.size();
}

} // namespace

std::unique_ptr<GraphicsBackend> createVulkanBackend() { return std::make_unique<VulkanBackend>(); }

NAMESPACE_END(krr)
