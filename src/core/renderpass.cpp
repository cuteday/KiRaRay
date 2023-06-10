#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

RenderTexture::RenderTexture(vkrhi::vulkan::IDevice *device, vkrhi::TextureHandle texture) :
	mTexture(texture) {
	auto desc		 = mTexture->getDesc();
	auto cudaHandler = std::make_unique<vkrhi::CuVkHandler>(device);
	mCudaSurface = cudaHandler->mapVulkanTextureToCudaSurface(mTexture, cudaArrayColorAttachment);
}

RenderTexture::SharedPtr RenderTexture::create(vkrhi::vulkan::IDevice *device, const Vector2i size,
											   vkrhi::Format format, const std::string name) {
	auto textureDesc = getVulkanDesc(size, format, name);
	return std::make_shared<RenderTexture>(device, device->createTexture(textureDesc));
}

void RenderTarget::resize(const Vector2i size) {

}

RenderContext::RenderContext(nvrhi::vulkan::Device* device,
							 nvrhi::CommandListHandle commandList) :
	mDevice(device),
	mCommandList(commandList),
	mCudaHandler(std::make_unique<vkrhi::CuVkHandler>(device)) {
	mCudaSemaphore = mCudaHandler->createCuVkSemaphore(true);
	mVulkanSemaphore = mCudaHandler->createCuVkSemaphore(true);
}

RenderContext::~RenderContext() {}

void RenderContext::resize(Vector2i size) { mRenderTarget->resize(size); }

void RenderContext::sychronizeCuda() {
	uint64_t waitValue = mDevice->getQueue(vkrhi::CommandQueue::Graphics)->getLastSubmittedID();
	cudaExternalSemaphore_t waitSemaphores[] = {mVulkanSemaphore};
	mCudaHandler->cudaWaitExternalSemaphore(mCudaStream, waitValue, waitSemaphores);
}

void RenderContext::sychronizeVulkan() {
	mCudaHandler->cudaSignalExternalSemaphore(mCudaStream, ++mCudaSemaphoreValue,
											  &mCudaSemaphore.cuda());
	mDevice->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, mCudaSemaphore,
								   mCudaSemaphoreValue);
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

