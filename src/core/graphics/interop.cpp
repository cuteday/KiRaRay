#include "interop.h"
#include "util/check.h"
#include <cuda_runtime.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace krr {
namespace {

void closeSharedHandle(nvrhi::IResource *resource) {
	if (!resource) return;
	auto handle = resource->getNativeObject(nvrhi::ObjectTypes::SharedHandle);
#ifdef _WIN32
	if (handle.pointer) CloseHandle(handle.pointer);
#else
	if (handle.integer != uint64_t(-1)) close(int(handle.integer));
#endif
}

cudaExternalMemory_t importMemory(nvrhi::IDevice *device, nvrhi::IResource *resource, uint64_t size) {
	if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
		return detail::importVulkanMemory(device, resource, size);
#ifdef KRR_ENABLE_D3D12
	if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
		return detail::importD3D12Memory(device, resource);
#endif
	throw std::runtime_error("Unsupported graphics API for CUDA sharing.");
}

cudaChannelFormatDesc channelFormat(nvrhi::Format format) {
	switch (format) {
	case nvrhi::Format::R32_FLOAT: return cudaCreateChannelDesc<float>();
	case nvrhi::Format::RG32_FLOAT: return cudaCreateChannelDesc<float2>();
	case nvrhi::Format::RGBA32_FLOAT: return cudaCreateChannelDesc<float4>();
	default: throw std::invalid_argument("CUDA render textures require R32, RG32, or RGBA32 float format.");
	}
}

}

struct CudaTextureMapping::Impl {
	nvrhi::TextureHandle texture;
	cudaExternalMemory_t memory{};
	cudaMipmappedArray_t array{};
	cudaSurfaceObject_t surface{};

	~Impl() {
		if (surface) cudaDestroySurfaceObject(surface);
		if (array) cudaFreeMipmappedArray(array);
		if (memory) cudaDestroyExternalMemory(memory);
		closeSharedHandle(texture);
	}
};

CudaTextureMapping::CudaTextureMapping(nvrhi::IDevice *device, nvrhi::TextureDesc desc) :
	mImpl(std::make_unique<Impl>()) {
	if (desc.dimension != nvrhi::TextureDimension::Texture2D || desc.arraySize != 1 ||
		desc.mipLevels != 1 || desc.sampleCount != 1 || desc.isVirtual || desc.isTiled)
		throw std::invalid_argument("CUDA render textures must be non-virtual, single-level 2D textures.");
	auto format = channelFormat(desc.format);
	desc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;
	desc.initialState = device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN ?
		nvrhi::ResourceStates::UnorderedAccess : nvrhi::ResourceStates::Common;
	desc.keepInitialState = true;
	desc.isUAV = true;
	mImpl->texture = device->createTexture(desc);
	if (!mImpl->texture) throw std::runtime_error("Could not create a shared render texture.");
	mImpl->memory = importMemory(device, mImpl->texture,
		device->getTextureMemoryRequirements(mImpl->texture).size);
	cudaExternalMemoryMipmappedArrayDesc mapping{};
	mapping.formatDesc = format;
	mapping.extent = make_cudaExtent(desc.width, desc.height, 0);
	mapping.numLevels = 1;
	mapping.flags = cudaArraySurfaceLoadStore | (desc.isRenderTarget ? cudaArrayColorAttachment : 0);
	CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mImpl->array, mImpl->memory, &mapping));
	cudaArray_t array{};
	CUDA_CHECK(cudaGetMipmappedArrayLevel(&array, mImpl->array, 0));
	cudaResourceDesc resource{};
	resource.resType = cudaResourceTypeArray;
	resource.res.array.array = array;
	CUDA_CHECK(cudaCreateSurfaceObject(&mImpl->surface, &resource));
}

CudaTextureMapping::~CudaTextureMapping() = default;
CudaTextureMapping::CudaTextureMapping(CudaTextureMapping &&) noexcept = default;
CudaTextureMapping &CudaTextureMapping::operator=(CudaTextureMapping &&) noexcept = default;
nvrhi::ITexture *CudaTextureMapping::getTexture() const { return mImpl ? mImpl->texture.Get() : nullptr; }
cudaSurfaceObject_t CudaTextureMapping::getSurface() const { return mImpl ? mImpl->surface : 0; }

struct CudaBufferMapping::Impl {
	nvrhi::BufferHandle buffer;
	cudaExternalMemory_t memory{};
	void *pointer{};

	~Impl() {
		if (pointer) cudaFree(pointer);
		if (memory) cudaDestroyExternalMemory(memory);
		closeSharedHandle(buffer);
	}
};

CudaBufferMapping::CudaBufferMapping(nvrhi::IDevice *device, nvrhi::BufferDesc desc) :
	mImpl(std::make_unique<Impl>()) {
	if (!desc.byteSize || desc.isVirtual || desc.cpuAccess != nvrhi::CpuAccessMode::None)
		throw std::invalid_argument("CUDA shared buffers must have a nonzero size and device-local allocation.");
	desc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;
	desc.initialState = nvrhi::ResourceStates::Common;
	desc.keepInitialState = true;
	mImpl->buffer = device->createBuffer(desc);
	if (!mImpl->buffer) throw std::runtime_error("Could not create a shared buffer.");
	mImpl->memory = importMemory(device, mImpl->buffer,
		device->getBufferMemoryRequirements(mImpl->buffer).size);
	cudaExternalMemoryBufferDesc mapping{};
	mapping.size = desc.byteSize;
	CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&mImpl->pointer, mImpl->memory, &mapping));
}

CudaBufferMapping::~CudaBufferMapping() = default;
CudaBufferMapping::CudaBufferMapping(CudaBufferMapping &&) noexcept = default;
CudaBufferMapping &CudaBufferMapping::operator=(CudaBufferMapping &&) noexcept = default;
nvrhi::IBuffer *CudaBufferMapping::getBuffer() const { return mImpl ? mImpl->buffer.Get() : nullptr; }
void *CudaBufferMapping::getPointer() const { return mImpl ? mImpl->pointer : nullptr; }

GraphicsInterop::GraphicsInterop(nvrhi::IDevice *device) : mDevice(device) {
	mPrepareCommand = device->createCommandList();
	mHandoffCommand = device->createCommandList();
	if (!mPrepareCommand || !mHandoffCommand)
		throw std::runtime_error("Could not create graphics interop command lists.");
}

GraphicsInterop::~GraphicsInterop() {
	if (mGraphicsReady) cudaDestroyExternalSemaphore(mGraphicsReady);
	if (mCudaReady) cudaDestroyExternalSemaphore(mCudaReady);
}

void GraphicsInterop::beginCuda(const std::vector<nvrhi::ITexture *> &textures,
	const std::vector<nvrhi::IBuffer *> &buffers, cudaStream_t stream) {
	mPrepareCommand->open();
	for (auto *texture : textures)
		mPrepareCommand->setTextureState(texture, nvrhi::AllSubresources, texture->getDesc().initialState);
	for (auto *buffer : buffers)
		mPrepareCommand->setBufferState(buffer, nvrhi::ResourceStates::Common);
	mPrepareCommand->close();
	// Keep NVRHI's closing barriers before the external ownership release.
	mHandoffCommand->open();
	ownership(mHandoffCommand, textures, buffers, true);
	mHandoffCommand->close();
	nvrhi::ICommandList *commands[] = {mPrepareCommand, mHandoffCommand};
	signalGraphics(++mGraphicsValue, commands, std::size(commands));
	cudaExternalSemaphoreWaitParams wait{};
	wait.params.fence.value = mGraphicsValue;
	CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&mGraphicsReady, &wait, 1, stream));
}

void GraphicsInterop::endCuda(const std::vector<nvrhi::ITexture *> &textures,
	const std::vector<nvrhi::IBuffer *> &buffers, cudaStream_t stream) {
	cudaExternalSemaphoreSignalParams signal{};
	signal.params.fence.value = ++mCudaValue;
	CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&mCudaReady, &signal, 1, stream));
	mHandoffCommand->open();
	ownership(mHandoffCommand, textures, buffers, false);
	mHandoffCommand->close();
	waitGraphics(mCudaValue);
	mDevice->executeCommandList(mHandoffCommand);
}

}
