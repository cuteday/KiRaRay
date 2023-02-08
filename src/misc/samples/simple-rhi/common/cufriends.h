#pragma once
#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <aclapi.h>
#include <dxgi1_2.h>
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
#ifdef _WIN64
class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES *operator&();
	~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
	m_winPSecurityDescriptor =
		(PSECURITY_DESCRIPTOR) calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
	if (!m_winPSecurityDescriptor) {
		throw std::runtime_error("Failed to allocate memory for security descriptor");
	}

	PSID *ppSID = (PSID *) ((PBYTE) m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL *ppACL = (PACL *) ((PBYTE) ppSID + sizeof(PSID *));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0,
							 ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode		= SET_ACCESS;
	explicitAccess.grfInheritance		= INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm	= TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType	= TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName	= (LPTSTR) *ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength				 = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle		 = TRUE;
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() { return &m_winSecurityAttributes; }

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
	PSID *ppSID = (PSID *) ((PBYTE) m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL *ppACL = (PACL *) ((PBYTE) ppSID + sizeof(PSID *));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}
#endif /* _WIN64 */

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
							   VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if (typeFilter & (1 << i) &&
			(memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	return ~0;
}

VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
#ifdef _WIN64
	return IsWindows8Point1OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
									   : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
	return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}	

VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType() {
#ifdef _WIN64
	return IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
								 : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
	return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
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
	VkDeviceMemory vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType) {
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

void createExternalBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
							VkDeviceSize size, VkBufferUsageFlags usage,
							VkMemoryPropertyFlags properties,
							VkExternalMemoryHandleTypeFlagsKHR extMemHandleType,
							VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType			  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size				  = size;
	bufferInfo.usage			  = usage;
	bufferInfo.sharingMode		  = VK_SHARING_MODE_EXCLUSIVE;

	VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
	externalMemoryBufferInfo.sType		 = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	externalMemoryBufferInfo.handleTypes = extMemHandleType;
	bufferInfo.pNext					 = &externalMemoryBufferInfo;

	if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to create buffer!");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

#ifdef _WIN64
	WindowsSecurityAttributes winSecurityAttributes;

	VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
	vulkanExportMemoryWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	vulkanExportMemoryWin32HandleInfoKHR.pNext		 = NULL;
	vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
	vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
		DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR) NULL;
#endif /* _WIN64 */
	VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
	vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
	vulkanExportMemoryAllocateInfoKHR.pNext =
		extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
			? &vulkanExportMemoryWin32HandleInfoKHR
			: NULL;
	vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
#else
	vulkanExportMemoryAllocateInfoKHR.pNext		  = NULL;
	vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType				   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext				   = &vulkanExportMemoryAllocateInfoKHR;
	allocInfo.allocationSize	   = memRequirements.size;
	allocInfo.memoryTypeIndex =
		findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate external buffer memory!");
	}

	vkBindBufferMemory(device, buffer, bufferMemory, 0);
}


void createExternalSemaphore(VkDevice device, VkSemaphore &semaphore,
								VkExternalSemaphoreHandleTypeFlagBits handleType) {
	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType					= VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
	exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

#ifdef _VK_TIMELINE_SEMAPHORE
	VkSemaphoreTypeCreateInfo timelineCreateInfo;
	timelineCreateInfo.sType		 = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	timelineCreateInfo.pNext		 = NULL;
	timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	timelineCreateInfo.initialValue	 = 0;
	exportSemaphoreCreateInfo.pNext	 = &timelineCreateInfo;
#else
	exportSemaphoreCreateInfo.pNext = NULL;
#endif /* _VK_TIMELINE_SEMAPHORE */
	exportSemaphoreCreateInfo.handleTypes = handleType;
	semaphoreInfo.pNext					  = &exportSemaphoreCreateInfo;

	if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
		throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
	}
}


void importExternalBuffer(void *handle, VkDevice device, VkPhysicalDevice physicalDevice,
										 VkExternalMemoryHandleTypeFlagBits handleType, size_t size,
										 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
										 VkBuffer &buffer, VkDeviceMemory &memory) {
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType			  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size				  = size;
	bufferInfo.usage			  = usage;
	bufferInfo.sharingMode		  = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to create buffer!");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

#ifdef _WIN64
	VkImportMemoryWin32HandleInfoKHR handleInfo = {};
	handleInfo.sType	  = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	handleInfo.pNext	  = NULL;
	handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	handleInfo.handle	  = handle;
	handleInfo.name		  = NULL;
#else
	VkImportMemoryFdInfoKHR handleInfo			  = {};
	handleInfo.sType							  = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
	handleInfo.pNext							  = NULL;
	handleInfo.fd								  = (int) (uintptr_t) handle;
	handleInfo.handleType						  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */

	VkMemoryAllocateInfo memAllocation = {};
	memAllocation.sType				   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllocation.pNext				   = (void *) &handleInfo;
	memAllocation.allocationSize	   = size;
	memAllocation.memoryTypeIndex =
		findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(device, &memAllocation, nullptr, &memory) != VK_SUCCESS) {
		throw std::runtime_error("Failed to import allocation!");
	}

	vkBindBufferMemory(device, buffer, memory, 0);
}
}

namespace cufriends::vkrhi {	// this is upon nvrhi.

using namespace cufriends;

class CudaVulkanFriend {
public:
	CudaVulkanFriend(nvrhi::DeviceHandle& device):
		m_device(reinterpret_cast<const nvrhi::vulkan::Device &>(device)) {}

	nvrhi::BufferHandle createExternalBuffer(nvrhi::BufferDesc desc) {

		
	
	}

	void importVulkanBufferToCuda(void **cudaPtr, cudaExternalMemory_t &cudaMem,
								  const nvrhi::BufferHandle &buffer) {
		
		auto *vk_buffer = dynamic_cast<nvrhi::vulkan::Buffer*>(buffer.Get());
		//importCudaExternalMemory(cudaPtr, cudaMem, vk_buffer->memory, vk_buffer->desc.byteSize,
		//						 cufriends::getDefaultMemHandleType());
	}

	void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem, 
									vk::DeviceMemory vkMemory, vk::DeviceSize size,
								  vk::ExternalMemoryHandleTypeFlagBits handleType) {
		const nvrhi::vulkan::VulkanContext &context = m_device.getContext();
		cufriends::importCudaExternalMemory(cudaPtr, cudaMem, context.device, vkMemory, size,
											VkExternalMemoryHandleTypeFlagBits(handleType));
	}

private:
	const nvrhi::vulkan::Device &m_device;
	
};
	
}

KRR_NAMESPACE_END