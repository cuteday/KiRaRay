#pragma once

#include <cuda_runtime_api.h>
#include <nvrhi/nvrhi.h>
#include <memory>
#include <vector>

namespace krr {

// Finish graphics and CUDA work before releasing a mapping.
class CudaTextureMapping {
public:
	CudaTextureMapping(nvrhi::IDevice *device, nvrhi::TextureDesc desc);
	~CudaTextureMapping();
	CudaTextureMapping(CudaTextureMapping &&) noexcept;
	CudaTextureMapping &operator=(CudaTextureMapping &&) noexcept;
	CudaTextureMapping(const CudaTextureMapping &) = delete;
	CudaTextureMapping &operator=(const CudaTextureMapping &) = delete;

	nvrhi::ITexture *getTexture() const;
	cudaSurfaceObject_t getSurface() const;

private:
	struct Impl;
	std::unique_ptr<Impl> mImpl;
};

class CudaBufferMapping {
public:
	CudaBufferMapping(nvrhi::IDevice *device, nvrhi::BufferDesc desc);
	~CudaBufferMapping();
	CudaBufferMapping(CudaBufferMapping &&) noexcept;
	CudaBufferMapping &operator=(CudaBufferMapping &&) noexcept;
	CudaBufferMapping(const CudaBufferMapping &) = delete;
	CudaBufferMapping &operator=(const CudaBufferMapping &) = delete;

	nvrhi::IBuffer *getBuffer() const;
	void *getPointer() const;

private:
	struct Impl;
	std::unique_ptr<Impl> mImpl;
};

class GraphicsInterop {
public:
	explicit GraphicsInterop(nvrhi::IDevice *device);
	virtual ~GraphicsInterop();
	GraphicsInterop(const GraphicsInterop &) = delete;
	GraphicsInterop &operator=(const GraphicsInterop &) = delete;

	void beginCuda(const std::vector<nvrhi::ITexture *> &textures,
		const std::vector<nvrhi::IBuffer *> &buffers, cudaStream_t stream);
	void endCuda(const std::vector<nvrhi::ITexture *> &textures,
		const std::vector<nvrhi::IBuffer *> &buffers, cudaStream_t stream);

protected:
	virtual void ownership(nvrhi::ICommandList *command,
		const std::vector<nvrhi::ITexture *> &textures,
		const std::vector<nvrhi::IBuffer *> &buffers, bool toCuda) = 0;
	virtual void signalGraphics(uint64_t value, nvrhi::ICommandList *const *commands, size_t count) = 0;
	virtual void waitGraphics(uint64_t value) = 0;

	nvrhi::DeviceHandle mDevice;
	cudaExternalSemaphore_t mGraphicsReady{};
	cudaExternalSemaphore_t mCudaReady{};

private:
	nvrhi::CommandListHandle mPrepareCommand;
	nvrhi::CommandListHandle mHandoffCommand;
	uint64_t mGraphicsValue{};
	uint64_t mCudaValue{};
};

std::unique_ptr<GraphicsInterop> createVulkanInterop(nvrhi::IDevice *device, uint32_t queueFamily);
#ifdef KRR_ENABLE_D3D12
std::unique_ptr<GraphicsInterop> createD3D12Interop(nvrhi::IDevice *device);
#endif

namespace detail {
cudaExternalMemory_t importVulkanMemory(nvrhi::IDevice *device, nvrhi::IResource *resource, uint64_t size);
#ifdef KRR_ENABLE_D3D12
cudaExternalMemory_t importD3D12Memory(nvrhi::IDevice *device, nvrhi::IResource *resource);
#endif
}

}
