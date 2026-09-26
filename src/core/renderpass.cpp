#include "renderpass.h"
#include "graphics/device.h"

NAMESPACE_BEGIN(krr)

DeviceManager* RenderPass::getDeviceManager() const {
	return mDeviceManager;
}

nvrhi::IDevice *RenderPass::getDevice() const {
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

