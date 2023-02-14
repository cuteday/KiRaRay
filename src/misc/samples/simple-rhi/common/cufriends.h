// https://github.com/Devsh-Graphics-Programming/Nabla/issues/395
// 
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
#include <nvrhi/vulkan/vulkan-texture.h>

#include <common.h>
#include <logger.h>
#include <util/check.h>

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
#endif /* _WIN64 */

static inline uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter,
							   vk::MemoryPropertyFlags properties) {
	vk::PhysicalDeviceMemoryProperties memProperties;
	physicalDevice.getMemoryProperties(&memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if (typeFilter & (1 << i) &&
			(memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	return ~0;
}

inline vk::ExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
#ifdef _WIN64
	return IsWindows8Point1OrGreater() ? vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
									   : vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt;
#else
	return vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif /* _WIN64 */
}	

inline vk::ExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType() {
#ifdef _WIN64
	return IsWindows8OrGreater() ? vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32
								 : vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt;
#else
	return vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif /* _WIN64 */
}

inline void *getMemHandle(vk::Device device, vk::DeviceMemory memory,
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
		(PFN_vkGetMemoryWin32HandleKHR) device.getProcAddr("vkGetMemoryWin32HandleKHR");
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
	vk::MemoryGetFdInfoKHR vkMemoryGetFdInfoKHR;
	vkMemoryGetFdInfoKHR.sType				  = vk::StructureType::eMemoryGetFdInfoKHR;
	vkMemoryGetFdInfoKHR.pNext				  = nullptr;
	vkMemoryGetFdInfoKHR.memory				  = memory;
	vkMemoryGetFdInfoKHR.handleType			  = vk::ExternalMemoryHandleTypeFlagBits(handleType);

	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
	fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) device.getProcAddr("vkGetMemoryFdKHR");
	if (!fpGetMemoryFdKHR) {
		throw std::runtime_error("Failed to retrieve vkGetMemoryFdKHR!");
	}
	device.getMemoryFdKHR(&vkMemoryGetFdInfoKHR, &fd);
	if (fpGetMemoryFdKHR(device, (const VkMemoryGetFdInfoKHR*) & vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
		throw std::runtime_error("Failed to retrieve handle for buffer!");
	}
	return (void *) (uintptr_t) fd;
#endif /* _WIN64 */
}

inline void *getSemaphoreHandle(vk::Device device, vk::Semaphore semaphore,
										vk::ExternalSemaphoreHandleTypeFlagBits handleType) {
#ifdef _WIN64
	HANDLE handle;

	VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
	semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	semaphoreGetWin32HandleInfoKHR.pNext = NULL;
	semaphoreGetWin32HandleInfoKHR.semaphore  = semaphore;
	semaphoreGetWin32HandleInfoKHR.handleType = VkExternalSemaphoreHandleTypeFlagBits(handleType);

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

inline void importCudaExternalSemaphore(cudaExternalSemaphore_t &cudaSem, vk::Device device,
	vk::Semaphore &vkSem, vk::ExternalSemaphoreHandleTypeFlagBits handleType) {
	cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

	if (handleType & vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	} else if (handleType & vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
	} else if (handleType & vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd) {
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	}
	else {
		throw std::runtime_error("Unknown handle type requested!");
	}

#ifdef _WIN64
	externalSemaphoreHandleDesc.handle.win32.handle =
		(HANDLE) getSemaphoreHandle(device, vkSem, vk::ExternalSemaphoreHandleTypeFlagBits(handleType));
#else
	externalSemaphoreHandleDesc.handle.fd =
		(int) (uintptr_t) getSemaphoreHandle(device, vkSem, handleType);
#endif

	externalSemaphoreHandleDesc.flags = 0;
	CUDA_CHECK(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
}

inline void createExternalSemaphore(vk::Device device, vk::Semaphore &semaphore,
								vk::ExternalSemaphoreHandleTypeFlagBits handleType) {
	vk::SemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType					= vk::StructureType::eSemaphoreCreateInfo;
	vk::ExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
	exportSemaphoreCreateInfo.sType = vk::StructureType::eExportSemaphoreCreateInfoKHR;
	exportSemaphoreCreateInfo.pNext = NULL;
	exportSemaphoreCreateInfo.handleTypes = handleType;
	semaphoreInfo.pNext					  = &exportSemaphoreCreateInfo;

	if (device.createSemaphore(& semaphoreInfo, nullptr, &semaphore) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
	}
}

inline void importExternalBuffer(void *handle, vk::Device device, vk::PhysicalDevice physicalDevice,
										 vk::ExternalMemoryHandleTypeFlagBits handleType, size_t size,
										 vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
										 vk::Buffer &buffer, vk::DeviceMemory &memory) {
	vk::BufferCreateInfo bufferInfo = {};
	bufferInfo.sType				= vk::StructureType::eBufferCreateInfo;			  
	bufferInfo.size					= size;
	bufferInfo.usage				= usage;
	bufferInfo.sharingMode			= vk::SharingMode::eExclusive;

	if (device.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create buffer!");
	}

	vk::MemoryRequirements memRequirements;
	device.getBufferMemoryRequirements(buffer, &memRequirements);
	
#ifdef _WIN64
	VkImportMemoryWin32HandleInfoKHR handleInfo = {};
	handleInfo.sType	  = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	handleInfo.pNext	  = NULL;
	handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	handleInfo.handle	  = handle;
	handleInfo.name		  = NULL;
#else
	vk::ImportMemoryFdInfoKHR handleInfo			  = {};
	handleInfo.sType							  = vk::StructureType::eImportMemoryFdInfoKHR;
	handleInfo.pNext							  = NULL;
	handleInfo.fd								  = (int) (uintptr_t) handle;
	handleInfo.handleType						  = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif /* _WIN64 */

	vk::MemoryAllocateInfo memAllocation = {};
	memAllocation.sType				   = vk::StructureType::eMemoryAllocateInfo;
	memAllocation.pNext				   = (void *) &handleInfo;
	memAllocation.allocationSize	   = size;
	memAllocation.memoryTypeIndex =
		findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	if (device.allocateMemory(& memAllocation, nullptr, &memory) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to import allocation!");
	}

	vkBindBufferMemory(device, buffer, memory, 0);
}
}

namespace vkrhi {	// this is upon nvrhi.

static vk::MemoryPropertyFlags pickBufferMemoryProperties(const nvrhi::BufferDesc &d) {
	vk::MemoryPropertyFlags flags{};

	switch (d.cpuAccess) {
		case nvrhi::CpuAccessMode::None:
			flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
			break;
		case nvrhi::CpuAccessMode::Read:
			flags =
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
			break;
		case nvrhi::CpuAccessMode::Write:
			flags = vk::MemoryPropertyFlagBits::eHostVisible;
			break;
	}

	return flags;
}

using namespace cufriends;

class CudaVulkanFriend {
public:
	using Buffer = nvrhi::vulkan::Buffer;

	CudaVulkanFriend(nvrhi::DeviceHandle device):
		m_device(*dynamic_cast<nvrhi::vulkan::Device*>(device.Get())),
		m_context(dynamic_cast<nvrhi::vulkan::Device*>(device.Get())->getContext()) {}

	vk::Result allocateExternalMemory(nvrhi::vulkan::MemoryResource* res,
		vk::MemoryRequirements memRequirements,
		vk::MemoryPropertyFlags memPropertyFlags,
		vk::ExternalMemoryHandleTypeFlags extMemHandleType,
		bool enableDeviceAddress = false) const {
		
		res->managed = true;

		// find a memory space that satisfies the requirements
		vk::PhysicalDeviceMemoryProperties memProperties;
		m_context.physicalDevice.getMemoryProperties(&memProperties);

		uint32_t memTypeIndex;
		for (memTypeIndex = 0; memTypeIndex < memProperties.memoryTypeCount; memTypeIndex++) {
			if ((memRequirements.memoryTypeBits & (1 << memTypeIndex)) &&
				((memProperties.memoryTypes[memTypeIndex].propertyFlags & memPropertyFlags) ==
				 memPropertyFlags)) {
				break;
			}
		}

		if (memTypeIndex == memProperties.memoryTypeCount) {
			// xxxnsubtil: this is incorrect; need better error reporting
			return vk::Result::eErrorOutOfDeviceMemory;
		}

		// allocate memory
		auto allocFlags = vk::MemoryAllocateFlagsInfo();
		if (enableDeviceAddress) allocFlags.flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;

		auto allocInfo = vk::MemoryAllocateInfo()
							 .setAllocationSize(memRequirements.size)
							 .setMemoryTypeIndex(memTypeIndex);
		
		/* external memory handle info */

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;
		VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
		vulkanExportMemoryWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		vulkanExportMemoryWin32HandleInfoKHR.pNext		 = nullptr;
		vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR) nullptr;
#endif /* _WIN64 */
		vk::ExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType = vk::StructureType::eExportMemoryAllocateInfoKHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext =
			extMemHandleType & vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
				? &vulkanExportMemoryWin32HandleInfoKHR : nullptr;
		vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = nullptr;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif /* _WIN64 */

		allocFlags.setPNext(&vulkanExportMemoryAllocateInfoKHR);
		allocInfo.setPNext(&allocFlags);

		return m_context.device.allocateMemory(&allocInfo, m_context.allocationCallbacks,
											   &res->memory);
	}

	vk::Result allocateExternalBufferMemory(
			nvrhi::vulkan::Buffer *buffer,
			vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType,
			bool enableDeviceAddress = false) const {
		vk::MemoryRequirements memRequirements;
		m_context.device.getBufferMemoryRequirements(buffer->buffer, &memRequirements);

		// allocate memory
		const vk::Result res = allocateExternalMemory(
			buffer, 
			memRequirements, 
			vkrhi::pickBufferMemoryProperties(buffer->desc), 
			extMemHandleType,
			enableDeviceAddress);
		CHECK_VK_RETURN(res)
			
		m_context.device.bindBufferMemory(buffer->buffer, buffer->memory, 0);
		return vk::Result::eSuccess;
	}

	vk::Result allocateExternalTextureMemory(nvrhi::vulkan::Texture *texture, 
			vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType) {
		// grab the image memory requirements
		vk::MemoryRequirements memRequirements;
		m_context.device.getImageMemoryRequirements(texture->image, &memRequirements);

		Log(Debug, "Allocating external image memory of size %lld", memRequirements.size);
		// allocate memory
		const vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal;
		const vk::Result res = allocateExternalMemory(texture, memRequirements, memProperties, extMemHandleType);
		CHECK_VK_RETURN(res)
	
		m_context.device.bindImageMemory(texture->image, texture->memory, 0);
		return vk::Result::eSuccess;
	}

	/*	1. specify ExternalMemoryBufferCreateInfo when creating buffer;
		2. specify ExportMemoryAllocateInfoKHR when creating memory. */
	nvrhi::BufferHandle createExternalBuffer(nvrhi::BufferDesc desc, 
		vk::ExternalMemoryHandleTypeFlags extMemHandleType) {
		if (desc.isVolatile && desc.maxVersions == 0) return nullptr;
		if (desc.isVolatile && !desc.isConstantBuffer) return nullptr;
		if (desc.byteSize == 0) return nullptr;

		auto *buffer = new nvrhi::vulkan::Buffer(m_context, m_device.getAllocator());
		buffer->desc   = desc;

		vk::BufferUsageFlags usageFlags =
			vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
		if (desc.isVertexBuffer) usageFlags |= vk::BufferUsageFlagBits::eVertexBuffer;
		if (desc.isIndexBuffer) usageFlags |= vk::BufferUsageFlagBits::eIndexBuffer;
		if (desc.isDrawIndirectArgs) usageFlags |= vk::BufferUsageFlagBits::eIndirectBuffer;
		if (desc.isConstantBuffer) usageFlags |= vk::BufferUsageFlagBits::eUniformBuffer;
		if (desc.structStride != 0 || desc.canHaveUAVs || desc.canHaveRawViews)
			usageFlags |= vk::BufferUsageFlagBits::eStorageBuffer;
		if (desc.canHaveTypedViews) usageFlags |= vk::BufferUsageFlagBits::eUniformTexelBuffer;
		if (desc.canHaveTypedViews && desc.canHaveUAVs)
			usageFlags |= vk::BufferUsageFlagBits::eStorageTexelBuffer;
		if (desc.isAccelStructBuildInput)
			usageFlags |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		if (desc.isAccelStructStorage)
			usageFlags |= vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR;
		if (m_context.extensions.buffer_device_address)
			usageFlags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;

		uint64_t size = desc.byteSize;

		if (desc.isVolatile) {
			assert(!desc.isVirtual);
			uint64_t alignment =
				m_context.physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
			uint64_t atomSize = m_context.physicalDeviceProperties.limits.nonCoherentAtomSize;
			alignment		  = std::max(alignment, atomSize);
			assert((alignment & (alignment - 1)) == 0); // check if it's a power of 2
			size				  = (size + alignment - 1) & ~(alignment - 1);
			buffer->desc.byteSize = size;
			size *= desc.maxVersions;
			buffer->versionTracking.resize(desc.maxVersions);
			std::fill(buffer->versionTracking.begin(), buffer->versionTracking.end(), 0);

			buffer->desc.cpuAccess = nvrhi::CpuAccessMode::Write; // to get the right memory type allocated
		} else if (desc.byteSize < 65536) {
			size += size % 4;
		}

		auto externalMemoryBufferInfo =
			vk::ExternalMemoryBufferCreateInfo().setHandleTypes(extMemHandleType);
		auto bufferInfo = vk::BufferCreateInfo()
							  .setSize(size)
							  .setUsage(usageFlags)
							  .setSharingMode(vk::SharingMode::eExclusive)
							  .setPNext(&externalMemoryBufferInfo);

		vk::Result res = m_context.device.createBuffer(&bufferInfo, m_context.allocationCallbacks,
													   &buffer->buffer);
		CHECK_VK_FAIL(res);
		m_context.nameVKObject(VkBuffer(buffer->buffer), vk::DebugReportObjectTypeEXT::eBuffer,
							   desc.debugName.c_str());

		if (!desc.isVirtual) {
			res = allocateExternalBufferMemory(
				buffer, 
				extMemHandleType,
				(usageFlags & vk::BufferUsageFlagBits::eShaderDeviceAddress) != vk::BufferUsageFlags(0));
			CHECK_VK_FAIL(res)
			m_context.nameVKObject(buffer->memory, vk::DebugReportObjectTypeEXT::eDeviceMemory,
								   desc.debugName.c_str());
			if (desc.isVolatile) {
				buffer->mappedMemory = m_context.device.mapMemory(buffer->memory, 0, size);
				assert(buffer->mappedMemory);
			}

			if (m_context.extensions.buffer_device_address) {
				auto addressInfo = vk::BufferDeviceAddressInfo().setBuffer(buffer->buffer);
				buffer->deviceAddress = m_context.device.getBufferAddress(addressInfo);
			}
		}
		return nvrhi::BufferHandle::Create(buffer);
	}

	nvrhi::TextureHandle createExternalTexture(nvrhi::TextureDesc desc,
						  vk::ExternalMemoryHandleTypeFlags extMemHandleType) {
		nvrhi::vulkan::Texture *texture =
			new nvrhi::vulkan::Texture(m_context, m_device.getAllocator());
		assert(texture);
		
	 	nvrhi::vulkan::fillTextureInfo(texture, desc);
		auto externalMemoryImageInfo =
			vk::ExternalMemoryImageCreateInfo().setHandleTypes(extMemHandleType);
		texture->imageInfo.setPNext(&externalMemoryImageInfo);

		vk::Result res = m_context.device.createImage(
			&texture->imageInfo, m_context.allocationCallbacks, &texture->image);
		ASSERT_VK_OK(res);
		CHECK_VK_FAIL(res)

		m_context.nameVKObject(texture->image, vk::DebugReportObjectTypeEXT::eImage,
							   desc.debugName.c_str());

		if (!desc.isVirtual) {
			res = allocateExternalTextureMemory(texture, extMemHandleType);
			ASSERT_VK_OK(res);
			CHECK_VK_FAIL(res)
			m_context.nameVKObject(texture->memory, vk::DebugReportObjectTypeEXT::eDeviceMemory,
								   desc.debugName.c_str());
		}
		return nvrhi::TextureHandle::Create(texture);
	}
	
	void importVulkanMemoryToCuda(cudaExternalMemory_t &cudaMem, 
		vk::DeviceMemory memory, vk::DeviceSize size) {
			cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

			vk::Device device = m_context.device;
			vk::ExternalMemoryHandleTypeFlagBits handleType = getDefaultMemHandleType();

			if (handleType & vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32) {
				externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
			} else if (handleType & vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt) {
				externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
			} else if (handleType & vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd) {
				externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
			} else {
				throw std::runtime_error("Unknown handle type requested!");
			}

			externalMemoryHandleDesc.size = size;

#ifdef _WIN64
			externalMemoryHandleDesc.handle.win32.handle = (HANDLE) getMemHandle(
				device, memory, VkExternalMemoryHandleTypeFlagBits(handleType));
#else
			externalMemoryHandleDesc.handle.fd = (int) (uintptr_t) getMemHandle(
				device, memory, VkExternalMemoryHandleTypeFlagBits(handleType));
#endif

			CUDA_CHECK(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));
	}

	void importVulkanBufferToCudaPtr(void **cudaPtr, cudaExternalMemory_t &cudaMem,
								  const nvrhi::BufferHandle &buffer) {
		auto *vk_buffer = dynamic_cast<nvrhi::vulkan::Buffer*>(buffer.Get());
		importVulkanMemoryToCuda(cudaMem, vk_buffer->memory, vk_buffer->desc.byteSize);
		
		cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
		externalMemBufferDesc.offset					   = 0;
		externalMemBufferDesc.size						   = vk_buffer->desc.byteSize;
		externalMemBufferDesc.flags						   = 0;

		CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
	}

	cudaMipmappedArray_t importVulkanTextureToCudaMipmappedArray(nvrhi::TextureHandle tex, 
		uint32_t cudaUsageFlags) {
		auto getChannelBits = [](nvrhi::FormatInfo format) -> std::array<size_t, 4> {
			size_t n_channels =
				size_t(format.hasRed) + format.hasBlue + format.hasGreen + format.hasAlpha;
			size_t bits_per_channel = format.bytesPerBlock * 8 / n_channels;
			return {bits_per_channel * format.hasRed, bits_per_channel * format.hasGreen,
					bits_per_channel * format.hasBlue, bits_per_channel * format.hasAlpha};
		};

		cudaExternalMemory_t cudaMem{};
		auto *vk_texture = dynamic_cast<nvrhi::vulkan::Texture *>(tex.Get());
		const nvrhi::TextureDesc& texture_desc = vk_texture->getDesc();
		importVulkanMemoryToCuda(cudaMem, vk_texture->memory,
								 m_device.getTextureMemoryRequirements(tex).size);
		Log(Debug, "Importing a vulkan texture (size %lld) to cuda external memory",
			m_device.getTextureMemoryRequirements(tex).size);

		nvrhi::FormatInfo formatInfo = nvrhi::getFormatInfo(texture_desc.format);
		auto numChannelBits			 = getChannelBits(formatInfo);
		cudaMipmappedArray_t mipmappedArray;
		cudaExternalMemoryMipmappedArrayDesc mipDesc;
		memset(&mipDesc, 0, sizeof(mipDesc));
		
		mipDesc.formatDesc.x = numChannelBits[0];
		mipDesc.formatDesc.y = numChannelBits[1];
		mipDesc.formatDesc.z = numChannelBits[2];
		mipDesc.formatDesc.w = numChannelBits[3];
		mipDesc.formatDesc.f  = formatInfo.kind == nvrhi::FormatKind::Float
									? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
		
		mipDesc.extent.depth  = texture_desc.depth;
		mipDesc.extent.width  = texture_desc.width;
		mipDesc.extent.height = texture_desc.height;
		mipDesc.flags		  = cudaUsageFlags;
		mipDesc.numLevels	  = texture_desc.mipLevels;
		mipDesc.offset		  = 0;

		CUDA_CHECK(
			cudaExternalMemoryGetMappedMipmappedArray(&mipmappedArray, cudaMem, &mipDesc));
		return mipmappedArray;
	}
	
	cudaSurfaceObject_t mapVulkanTextureToCudaSurface(nvrhi::TextureHandle texture,
													  uint32_t cudaUsageFlags) {
		cudaMipmappedArray_t mipmappedArray =
			importVulkanTextureToCudaMipmappedArray(texture, cudaUsageFlags);
		
		cudaArray_t array;
		CUDA_CHECK(cudaGetMipmappedArrayLevel(&array, mipmappedArray, 0));

		// Create cudaSurfObject_t from CUDA array
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.res.array.array = array;
		resDesc.resType			= cudaResourceTypeArray;

		cudaSurfaceObject_t surface;
		CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));
		return surface;
	}

	void getDeviceUUID(uint8_t* uuid) {
		auto physicalDeviceIDProperties = vk::PhysicalDeviceIDProperties();
		auto physicalDeviceProperties2	= vk::PhysicalDeviceProperties2();

		physicalDeviceProperties2.pNext = &physicalDeviceIDProperties;
		m_context.physicalDevice.getProperties2(&physicalDeviceProperties2);
		memcpy(uuid, physicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);
	}

	int initCUDA() {
		int current_device	   = 0;
		int device_count	   = 0;
		int devices_prohibited = 0;

		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceCount(&device_count));

		if (device_count == 0) {
			Log(Error, "CUDA error: no devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}

		// Find the GPU which is selected by Vulkan
		uint8_t vkDeviceUUID[VK_UUID_SIZE];
		getDeviceUUID(vkDeviceUUID);
		while (current_device < device_count) {
			cudaGetDeviceProperties(&deviceProp, current_device);

			if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
				// Compare the cuda device UUID with vulkan UUID
				int ret = memcmp((void *) &deviceProp.uuid, vkDeviceUUID, VK_UUID_SIZE);
				if (ret == 0) {
					CUDA_CHECK(cudaSetDevice(current_device));
					CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, current_device));
					Log(Info, "GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
						   current_device, deviceProp.name, deviceProp.major, deviceProp.minor);

					return current_device;
				}

			} else {
				devices_prohibited++;
			}
			current_device++;
		}

		if (devices_prohibited == device_count) {
			Log(Error, "CUDA error:"
							" No Vulkan-CUDA Interop capable GPU found.\n");
			exit(EXIT_FAILURE);
		}

		return -1;
	}

private:
	nvrhi::vulkan::Device &m_device;
	const nvrhi::vulkan::VulkanContext &m_context;
};
	
}

KRR_NAMESPACE_END