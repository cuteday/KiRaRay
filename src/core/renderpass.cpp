#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

RenderContext::RenderContext(nvrhi::vulkan::DeviceHandle device,
						   nvrhi::CommandListHandle commandList) :
	mDevice(device), 
	mCommandList(commandList),
	mCudaHandler(std::make_unique<vkrhi::CuVkHandler>(device)) {
}

DeviceManager* RenderPass::getDeviceManager() const {
	return mDeviceManager;
}

vk::Device RenderPass::getVulkanNativeDevice() const {
	return mDeviceManager->GetNativeDevice();
}

vkrhi::vulkan::IDevice *RenderPass::getVulkanDevice() const {
	return mDeviceManager->GetDevice();
}

size_t RenderPass::getFrameIndex() const { 
	return mDeviceManager->GetFrameIndex(); 
}

KRR_NAMESPACE_END

