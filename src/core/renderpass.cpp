#include "renderpass.h"
#include "window.h"
#include "device/context.h"

NAMESPACE_BEGIN(krr)

RenderTexture::RenderTexture(vkrhi::IDevice *device, vkrhi::TextureHandle texture) :
	mTexture(texture) {
	auto desc		 = mTexture->getDesc();
	auto cudaHandler = std::make_unique<vkrhi::CuVkHandler>(device);
	mCudaSurface = cudaHandler->mapVulkanTextureToCudaSurface(mTexture, cudaArrayColorAttachment);
}

vkrhi::TextureDesc RenderTexture::getVulkanDesc(const Vector2i size, vkrhi::Format format,
										const std::string name) {
	vkrhi::TextureDesc textureDesc;
	textureDesc.width			 = size[0];
	textureDesc.height			 = size[1];
	textureDesc.format			 = format;
	textureDesc.debugName		 = name;
	textureDesc.initialState	 = nvrhi::ResourceStates::ShaderResource;
	textureDesc.keepInitialState = true;
	textureDesc.isRenderTarget	 = true;
	textureDesc.isUAV			 = true;
	textureDesc.sampleCount		 = 1;
	return textureDesc;
}

RenderTexture::SharedPtr RenderTexture::create(vkrhi::IDevice *device, const Vector2i size,
											   vkrhi::Format format, const std::string name) {
	auto textureDesc = getVulkanDesc(size, format, name);
	auto cudaHandler = std::make_unique<vkrhi::CuVkHandler>(device);
	return std::make_shared<RenderTexture>(device, cudaHandler->createExternalTexture(textureDesc));
}

void RenderTarget::resize(const Vector2i size) {
	mSize = size;
	if (size[0] * size[1] == 0) return;

	mColor = RenderTexture::create(mDevice, size, vkrhi::Format::RGBA32_FLOAT, "RGB Texture");

	if (mEnableDepth)
		mDepth = RenderTexture::create(mDevice, size, vkrhi::Format::R32_FLOAT, "Depth Texture");
	else mDepth.reset();
	if (mEnableDiffuse)
		mDiffuse = RenderTexture::create(mDevice, size, vkrhi::Format::RGB32_FLOAT, "Diffuse Texture");
	else mDiffuse.reset();
	if (mEnableSpecular)
		mSpecular = RenderTexture::create(mDevice, size, vkrhi::Format::RGB32_FLOAT, "Specular Texture");
	else mSpecular.reset();
	if (mEnableNormal)
		mNormal = RenderTexture::create(mDevice, size, vkrhi::Format::RGB32_FLOAT, "Normal Texture");
	else mNormal.reset();
	if (mEnableMotion)
		mMotion = RenderTexture::create(mDevice, size, vkrhi::Format::RGB32_FLOAT, "Motion Texture");
	else mMotion.reset();
	if (mEnableEmissive)
		mEmissive = RenderTexture::create(mDevice, size, vkrhi::Format::RGB32_FLOAT, "Emissive Texture");
	else mEmissive.reset();

	mFramebuffer =
		mDevice->createFramebuffer(vkrhi::FramebufferDesc().addColorAttachment(mColor->getVulkanTexture()));
}

RenderContext::RenderContext(nvrhi::IDevice* device) :
	mDevice(device),
	mCudaHandler(std::make_unique<vkrhi::CuVkHandler>(device)) {
	mRenderTarget	 = std::make_shared<RenderTarget>(device);
	mCommandList	 = mDevice->createCommandList();
	mCudaSemaphore	 = mCudaHandler->createCuVkSemaphore(true);
	mVulkanSemaphore = mCudaHandler->createCuVkSemaphore(true);
	mCudaStream		 = KRR_DEFAULT_STREAM;
}

RenderContext::~RenderContext() { 
	vk::Device device = mDevice->getNativeObject(nvrhi::ObjectTypes::VK_Device);
	device.waitIdle(); 
	device.destroySemaphore(mCudaSemaphore);
	//device.destroySemaphore(mVulkanSemaphore);	// [TODO] this is managed by RHI.
}

void RenderContext::setScene(Scene::SharedPtr scene) {
	mScene = scene;
}

void RenderContext::resize(Vector2i size) { mRenderTarget->resize(size); }

void RenderContext::sychronizeCuda() {
	auto *device	   = dynamic_cast<vkrhi::vulkan::Device *>(mDevice);
	uint64_t waitValue = device->getQueue(vkrhi::CommandQueue::Graphics)->getLastSubmittedID();
	cudaExternalSemaphore_t waitSemaphores[] = {mVulkanSemaphore};
	mCudaHandler->cudaWaitExternalSemaphore(mCudaStream, waitValue, waitSemaphores);
}

void RenderContext::sychronizeVulkan() {
	auto *device = dynamic_cast<vkrhi::vulkan::Device *>(mDevice);
	mCudaHandler->cudaSignalExternalSemaphore(mCudaStream, ++mCudaSemaphoreValue,
											  &mCudaSemaphore.cuda());
	device->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, mCudaSemaphore,
								   mCudaSemaphoreValue);
}

DeviceManager* RenderPass::getDeviceManager() const {
	return mDeviceManager;
}

vk::Device RenderPass::getVulkanNativeDevice() const {
	return mDeviceManager->getNativeDevice();
}

vkrhi::vulkan::IDevice *RenderPass::getVulkanDevice() const {
	return mDeviceManager->getDevice();
}

size_t RenderPass::getFrameIndex() const { 
	return mDeviceManager->getFrameIndex(); 
}

Vector2i RenderPass::getFrameSize() const {
	return mDeviceManager->getFrameSize();
}

NAMESPACE_END(krr)

