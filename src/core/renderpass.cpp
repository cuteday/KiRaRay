#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

vk::Device RenderPass::getVulkanDevice() const {
	return mDeviceManager->GetNativeDevice();
}

vkrhi::IDevice *RenderPass::getVulkanRhiDevice() const {
	return mDeviceManager->GetDevice();
}

size_t RenderPass::getFrameIndex() const { 
	return mDeviceManager->GetFrameIndex(); 
}

KRR_NAMESPACE_END

