#include "graphics/interop.h"
#include "util/check.h"
#include <nvrhi/vulkan.h>

#ifdef _WIN32
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

namespace krr {
namespace {

void checkVulkan(VkResult result) {
	if (result != VK_SUCCESS)
		throw std::runtime_error("Vulkan interop failed: " + std::to_string(result));
}

class VulkanInterop : public GraphicsInterop {
public:
	VulkanInterop(nvrhi::IDevice *device, uint32_t queueFamily) :
		GraphicsInterop(device), mQueueFamily(queueFamily) {
		mVulkan = device->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device);
		mNativeDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
		if (!mVulkan || !mNativeDevice) throw std::runtime_error("Vulkan device is unavailable for CUDA sharing.");
		try {
			createSemaphore(mGraphicsSemaphore, mGraphicsReady);
			createSemaphore(mCudaSemaphore, mCudaReady);
		} catch (...) {
			release();
			throw;
		}
	}

	~VulkanInterop() override { release(); }

private:
	void release() noexcept {
		if (mGraphicsReady) cudaDestroyExternalSemaphore(mGraphicsReady);
		if (mCudaReady) cudaDestroyExternalSemaphore(mCudaReady);
		mGraphicsReady = mCudaReady = nullptr;
		if (mGraphicsSemaphore) vkDestroySemaphore(mNativeDevice, mGraphicsSemaphore, nullptr);
		if (mCudaSemaphore) vkDestroySemaphore(mNativeDevice, mCudaSemaphore, nullptr);
		mGraphicsSemaphore = mCudaSemaphore = VK_NULL_HANDLE;
	}

	void createSemaphore(VkSemaphore &semaphore, cudaExternalSemaphore_t &imported) {
		VkExportSemaphoreCreateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
#ifdef _WIN32
		exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
		VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, &exportInfo};
		typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &typeInfo};
		checkVulkan(vkCreateSemaphore(mNativeDevice, &info, nullptr, &semaphore));
		cudaExternalSemaphoreHandleDesc cudaInfo{};
#ifdef _WIN32
		auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
			vkGetDeviceProcAddr(mNativeDevice, "vkGetSemaphoreWin32HandleKHR"));
		if (!getHandle) throw std::runtime_error("Vulkan Win32 semaphore sharing is unavailable.");
		VkSemaphoreGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
		handleInfo.semaphore = semaphore;
		handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		HANDLE handle{};
		checkVulkan(getHandle(mNativeDevice, &handleInfo, &handle));
		cudaInfo.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
		cudaInfo.handle.win32.handle = handle;
		auto result = cudaImportExternalSemaphore(&imported, &cudaInfo);
		CloseHandle(handle);
#else
		auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
			vkGetDeviceProcAddr(mNativeDevice, "vkGetSemaphoreFdKHR"));
		if (!getHandle) throw std::runtime_error("Vulkan file descriptor semaphore sharing is unavailable.");
		VkSemaphoreGetFdInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
		handleInfo.semaphore = semaphore;
		handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
		int handle = -1;
		checkVulkan(getHandle(mNativeDevice, &handleInfo, &handle));
		cudaInfo.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
		cudaInfo.handle.fd = handle;
		auto result = cudaImportExternalSemaphore(&imported, &cudaInfo);
		if (result != cudaSuccess) close(handle);
#endif
		CUDA_CHECK(result);
	}

	void ownership(nvrhi::ICommandList *command, const std::vector<nvrhi::ITexture *> &textures,
		const std::vector<nvrhi::IBuffer *> &buffers, bool toCuda) override {
		std::vector<VkImageMemoryBarrier> images;
		std::vector<VkBufferMemoryBarrier> memory;
		for (auto *texture : textures) {
			VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
			barrier.srcAccessMask = toCuda ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
			barrier.dstAccessMask = toCuda ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
			barrier.oldLayout = barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.srcQueueFamilyIndex = toCuda ? mQueueFamily : VK_QUEUE_FAMILY_EXTERNAL;
			barrier.dstQueueFamilyIndex = toCuda ? VK_QUEUE_FAMILY_EXTERNAL : mQueueFamily;
			barrier.image = texture->getNativeObject(nvrhi::ObjectTypes::VK_Image);
			barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			images.push_back(barrier);
		}
		for (auto *buffer : buffers) {
			VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
			barrier.srcAccessMask = toCuda ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
			barrier.dstAccessMask = toCuda ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
			barrier.srcQueueFamilyIndex = toCuda ? mQueueFamily : VK_QUEUE_FAMILY_EXTERNAL;
			barrier.dstQueueFamilyIndex = toCuda ? VK_QUEUE_FAMILY_EXTERNAL : mQueueFamily;
			barrier.buffer = buffer->getNativeObject(nvrhi::ObjectTypes::VK_Buffer);
			barrier.size = VK_WHOLE_SIZE;
			memory.push_back(barrier);
		}
		VkCommandBuffer native = command->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
		vkCmdPipelineBarrier(native, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, uint32_t(memory.size()), memory.data(), uint32_t(images.size()), images.data());
	}

	void signalGraphics(uint64_t value, nvrhi::ICommandList *const *commands, size_t count) override {
		mVulkan->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, mGraphicsSemaphore, value);
		mDevice->executeCommandLists(commands, count);
	}

	void waitGraphics(uint64_t value) override {
		mVulkan->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, mCudaSemaphore, value);
	}

	nvrhi::vulkan::IDevice *mVulkan{};
	VkDevice mNativeDevice{};
	uint32_t mQueueFamily{};
	VkSemaphore mGraphicsSemaphore{};
	VkSemaphore mCudaSemaphore{};
};

}

std::unique_ptr<GraphicsInterop> createVulkanInterop(nvrhi::IDevice *device, uint32_t queueFamily) {
	return std::make_unique<VulkanInterop>(device, queueFamily);
}

cudaExternalMemory_t detail::importVulkanMemory(nvrhi::IDevice *, nvrhi::IResource *resource, uint64_t size) {
	cudaExternalMemoryHandleDesc desc{};
	desc.size = size;
	desc.flags = cudaExternalMemoryDedicated;
	auto handle = resource->getNativeObject(nvrhi::ObjectTypes::SharedHandle);
#ifdef _WIN32
	if (!handle.pointer) throw std::runtime_error("Vulkan resource has no shared handle.");
	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	desc.handle.win32.handle = handle.pointer;
#else
	desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	desc.handle.fd = dup(int(handle.integer));
	if (desc.handle.fd < 0) throw std::runtime_error("Could not duplicate Vulkan shared memory descriptor.");
#endif
	cudaExternalMemory_t memory{};
	auto result = cudaImportExternalMemory(&memory, &desc);
#ifndef _WIN32
	if (result != cudaSuccess) close(desc.handle.fd);
#endif
	CUDA_CHECK(result);
	return memory;
}

}
