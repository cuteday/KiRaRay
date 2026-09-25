#include "renderpass.h"
#include "window.h"
#include "device/context.h"

NAMESPACE_BEGIN(krr)

RenderTexture::RenderTexture(vkrhi::IDevice *device, vkrhi::TextureHandle texture) :
	mTexture(texture) {
	auto cudaHandler = std::make_unique<vkrhi::CuVkHandler>(device);
	try {
		mCudaSurface = cudaHandler->mapVulkanTextureToCudaSurface(mTexture,
			cudaArrayColorAttachment, mCudaArray, mCudaMemory);
	} catch (...) {
		if (mCudaArray) cudaFreeMipmappedArray(mCudaArray);
		if (mCudaMemory) cudaDestroyExternalMemory(mCudaMemory);
		throw;
	}
}

RenderTexture::~RenderTexture() { 
	if (mCudaSurface) cudaDestroySurfaceObject(mCudaSurface);
	if (mCudaArray) cudaFreeMipmappedArray(mCudaArray);
	if (mCudaMemory) cudaDestroyExternalMemory(mCudaMemory);
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
	try {
		mVulkanSemaphore = mCudaHandler->createCuVkSemaphore(true);
	} catch (...) {
		cudaDestroyExternalSemaphore(mCudaSemaphore.cuda());
		auto nativeDevice = static_cast<vk::Device>(mDevice->getNativeObject(nvrhi::ObjectTypes::VK_Device));
		nativeDevice.destroySemaphore(mCudaSemaphore);
		throw;
	}
	mCudaStream		 = KRR_DEFAULT_STREAM;
}

RenderContext::~RenderContext() { 
	vk::Device device =
		static_cast<vk::Device>(mDevice->getNativeObject(nvrhi::ObjectTypes::VK_Device));
	cudaStreamSynchronize(mCudaStream);
	device.waitIdle();
	mRenderTarget.reset();
	cudaDestroyExternalSemaphore(mCudaSemaphore.cuda());
	cudaDestroyExternalSemaphore(mVulkanSemaphore.cuda());
	device.destroySemaphore(mCudaSemaphore);
	if (mOwnsVulkanSemaphore) device.destroySemaphore(mVulkanSemaphore);
}

void RenderContext::setScene(Scene::SharedPtr scene) {
	mScene = scene;
}

void RenderContext::resize(Vector2i size) {
	CUDA_CHECK(cudaStreamSynchronize(mCudaStream));
	mDevice->waitForIdle();
	mRenderTarget->resize(size);
}

void RenderContext::clear() {
	sychronizeVulkan();
	mCommandList->open();
	mCommandList->clearTextureFloat(getColorTexture()->getVulkanTexture(),
		nvrhi::AllSubresources, nvrhi::Color(0.f));
	mCommandList->close();
	mDevice->executeCommandList(mCommandList);
}

std::vector<float> RenderContext::readback() {
	sychronizeVulkan();
	auto *texture = getColorTexture()->getVulkanTexture();
	const auto &desc = texture->getDesc();
	auto staging = mDevice->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);
	mCommandList->open();
	mCommandList->copyTexture(staging, nvrhi::TextureSlice(), texture, nvrhi::TextureSlice());
	mCommandList->close();
	mDevice->executeCommandList(mCommandList);
	mDevice->waitForIdle();
	std::vector<float> result(size_t(desc.width) * desc.height * 3);
	size_t pitch = 0;
	const auto *data = static_cast<const unsigned char *>(mDevice->mapStagingTexture(
		staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &pitch));
	if (!data) throw std::runtime_error("Could not map the rendered image.");
	for (uint32_t y = 0; y < desc.height; ++y) {
		const auto *row = reinterpret_cast<const float *>(data + y * pitch);
		for (uint32_t x = 0; x < desc.width; ++x)
			std::copy_n(row + x * 4, 3, result.data() + (size_t(y) * desc.width + x) * 3);
	}
	mDevice->unmapStagingTexture(staging);
	return result;
}

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
	// Submit the handoff even when no graphics pass follows it.
	device->executeCommandLists(nullptr, 0, nvrhi::CommandQueue::Graphics);
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

uint64_t RenderPass::getSeed() const {
	return mDeviceManager->getSeed();
}

bool RenderPass::isHeadless() const {
	return mDeviceManager->isHeadless();
}

Vector2i RenderPass::getFrameSize() const {
	return mDeviceManager->getFrameSize();
}

NAMESPACE_END(krr)

