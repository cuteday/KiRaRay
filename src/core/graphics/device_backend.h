#pragma once

#include "graphics/device.h"

NAMESPACE_BEGIN(krr)

class GraphicsInterop;

class GraphicsBackend {
public:
	virtual ~GraphicsBackend() = default;
	virtual void initialize(DeviceCreationParameters &params, GLFWwindow *window,
							nvrhi::IMessageCallback *callback) = 0;
	virtual nvrhi::IDevice *getDevice() const = 0;
	virtual const std::string &getRendererString() const = 0;
	virtual std::unique_ptr<GraphicsInterop> createInterop(nvrhi::IDevice *device) = 0;
	virtual void resizeSwapChain(const DeviceCreationParameters &params) = 0;
	virtual bool beginFrame() = 0;
	virtual void present() = 0;
	virtual nvrhi::ITexture *getBackBuffer(size_t index) const = 0;
	virtual size_t getBackBufferCount() const = 0;
	virtual size_t getCurrentBackBufferIndex() const = 0;
};

std::unique_ptr<GraphicsBackend> createVulkanBackend();
#if KRR_ENABLE_D3D12
std::unique_ptr<GraphicsBackend> createD3D12Backend();
#endif

NAMESPACE_END(krr)
