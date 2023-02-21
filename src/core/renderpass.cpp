#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

DeviceManager* RenderPass::getDeviceManager() const {
	return mDeviceManager;
}

vk::Device RenderPass::getVulkanNativeDevice() const {
	return mDeviceManager->GetNativeDevice();
}

vkrhi::IDevice *RenderPass::getVulkanDevice() const {
	return mDeviceManager->GetDevice();
}

size_t RenderPass::getFrameIndex() const { 
	return mDeviceManager->GetFrameIndex(); 
}

KRR_NAMESPACE_END

