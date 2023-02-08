#pragma once
#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <VersionHelpers.h>
#include <vulkan/vulkan_win32.h>
#endif /* _WIN64 */
#include <nvrhi/nvrhi.h>
#include <nvrhi/vulkan.h>
#include <nvrhi/vulkan/vulkan-backend.h>

#include <common.h>
#include <logger.h>

KRR_NAMESPACE_BEGIN

namespace cufriends {

VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
#ifdef _WIN64
	return IsWindows8Point1OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
									   : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
	return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}	

void* getMemHandle(VkDevice device, VkDeviceMemory memory,
								  VkExternalMemoryHandleTypeFlagBits handleType) {
#ifdef _WIN64
	HANDLE handle = 0;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
	vkMemoryGetWin32HandleInfoKHR.sType		 = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext		 = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory	 = memory;
	vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

	PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
	fpGetMemoryWin32HandleKHR =
		(PFN_vkGetMemoryWin32HandleKHR) vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
	if (!fpGetMemoryWin32HandleKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetMemoryWin32HandleKHR(device, &vkMemoryGetWin32HandleInfoKHR, &handle) !=
		VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}
	return (void *) handle;
#else
	int fd = -1;

	VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
	vkMemoryGetFdInfoKHR.sType				  = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	vkMemoryGetFdInfoKHR.pNext				  = NULL;
	vkMemoryGetFdInfoKHR.memory				  = memory;
	vkMemoryGetFdInfoKHR.handleType			  = handleType;

	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
	fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
	if (!fpGetMemoryFdKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}
	return (void *) (uintptr_t) fd;
#endif /* _WIN64 */
}

void *getSemaphoreHandle(VkDevice device, VkSemaphore semaphore,
										VkExternalSemaphoreHandleTypeFlagBits handleType) {
#ifdef _WIN64
	HANDLE handle;

	VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
	semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	semaphoreGetWin32HandleInfoKHR.pNext = NULL;
	semaphoreGetWin32HandleInfoKHR.semaphore  = semaphore;
	semaphoreGetWin32HandleInfoKHR.handleType = handleType;

	PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
	fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR) vkGetDeviceProcAddr(
		device, "vkGetSemaphoreWin32HandleKHR");
	if (!fpGetSemaphoreWin32HandleKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetSemaphoreWin32HandleKHR(device, &semaphoreGetWin32HandleInfoKHR, &handle) !=
		VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}

	return (void *) handle;
#else
	int fd;

	VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
	semaphoreGetFdInfoKHR.sType					  = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
	semaphoreGetFdInfoKHR.pNext					  = NULL;
	semaphoreGetFdInfoKHR.semaphore				  = semaphore;
	semaphoreGetFdInfoKHR.handleType			  = handleType;

	PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
	fpGetSemaphoreFdKHR =
		(PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
	if (!fpGetSemaphoreFdKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
	}
	if (fpGetSemaphoreFdKHR(device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}

	return (void *) (uintptr_t) fd;
#endif /* _WIN64 */
}


	
void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem, VkDevice device,
	VkDeviceMemory &vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType) {
	cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

	if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	} else {
		throw std::runtime_error("Unknown handle type requested!");
	}

	externalMemoryHandleDesc.size = size;

#ifdef _WIN64
	externalMemoryHandleDesc.handle.win32.handle = (HANDLE) getMemHandle(device, vkMem, handleType);
#else
	externalMemoryHandleDesc.handle.fd = (int) (uintptr_t) getMemHandle(device, vkMem, handleType);
#endif

	CUDA_CHECK(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

	cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
	externalMemBufferDesc.offset					   = 0;
	externalMemBufferDesc.size						   = size;
	externalMemBufferDesc.flags						   = 0;

	CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
}

void importCudaExternalSemaphore(cudaExternalSemaphore_t &cudaSem, VkDevice device,
	VkSemaphore &vkSem, VkExternalSemaphoreHandleTypeFlagBits handleType) {
	cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

#ifdef _VK_TIMELINE_SEMAPHORE
	if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
	}
#else
	if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
	} else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	}
#endif /* _VK_TIMELINE_SEMAPHORE */
	else {
		throw std::runtime_error("Unknown handle type requested!");
	}

#ifdef _WIN64
	externalSemaphoreHandleDesc.handle.win32.handle =
		(HANDLE) getSemaphoreHandle(device, vkSem, handleType);
#else
	externalSemaphoreHandleDesc.handle.fd =
		(int) (uintptr_t) getSemaphoreHandle(device, vkSem, handleType);
#endif

	externalSemaphoreHandleDesc.flags = 0;

	CUDA_CHECK(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
}
}

KRR_NAMESPACE_END