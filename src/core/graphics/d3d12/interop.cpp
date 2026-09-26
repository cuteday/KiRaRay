#include "graphics/interop.h"
#include "util/check.h"
#include <d3d12.h>
#include <wrl/client.h>

namespace krr {
namespace {

void checkD3D12(HRESULT result) {
	if (FAILED(result))
		throw std::runtime_error("D3D12 interop failed: " + std::to_string(result));
}

class D3D12Interop : public GraphicsInterop {
public:
	explicit D3D12Interop(nvrhi::IDevice *device) : GraphicsInterop(device) {
		mNativeDevice = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
		mQueue = device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);
		if (!mNativeDevice || !mQueue) throw std::runtime_error("D3D12 device is unavailable for CUDA sharing.");
		try {
			createFence(mGraphicsFence, mGraphicsReady);
			createFence(mCudaFence, mCudaReady);
		} catch (...) {
			release();
			throw;
		}
	}

	~D3D12Interop() override { release(); }

private:
	void release() noexcept {
		if (mGraphicsReady) cudaDestroyExternalSemaphore(mGraphicsReady);
		if (mCudaReady) cudaDestroyExternalSemaphore(mCudaReady);
		mGraphicsReady = mCudaReady = nullptr;
	}
	void createFence(Microsoft::WRL::ComPtr<ID3D12Fence> &fence, cudaExternalSemaphore_t &imported) {
		checkD3D12(mNativeDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)));
		HANDLE handle{};
		checkD3D12(mNativeDevice->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle));
		cudaExternalSemaphoreHandleDesc desc{};
		desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
		desc.handle.win32.handle = handle;
		auto result = cudaImportExternalSemaphore(&imported, &desc);
		CloseHandle(handle);
		CUDA_CHECK(result);
	}

	void ownership(nvrhi::ICommandList *, const std::vector<nvrhi::ITexture *> &,
		const std::vector<nvrhi::IBuffer *> &, bool) override {}

	void signalGraphics(uint64_t value, nvrhi::ICommandList *const *commands, size_t count) override {
		mDevice->executeCommandLists(commands, count);
		checkD3D12(mQueue->Signal(mGraphicsFence.Get(), value));
	}

	void waitGraphics(uint64_t value) override {
		checkD3D12(mQueue->Wait(mCudaFence.Get(), value));
	}

	ID3D12Device *mNativeDevice{};
	ID3D12CommandQueue *mQueue{};
	Microsoft::WRL::ComPtr<ID3D12Fence> mGraphicsFence;
	Microsoft::WRL::ComPtr<ID3D12Fence> mCudaFence;
};

}

std::unique_ptr<GraphicsInterop> createD3D12Interop(nvrhi::IDevice *device) {
	return std::make_unique<D3D12Interop>(device);
}

cudaExternalMemory_t detail::importD3D12Memory(nvrhi::IDevice *device, nvrhi::IResource *resource) {
	ID3D12Device *nativeDevice = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
	ID3D12Resource *nativeResource = resource->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
	if (!nativeDevice || !nativeResource) throw std::runtime_error("D3D12 shared resource is unavailable.");
	const auto resourceDesc = nativeResource->GetDesc();
	cudaExternalMemoryHandleDesc desc{};
	desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	desc.handle.win32.handle = resource->getNativeObject(nvrhi::ObjectTypes::SharedHandle);
	desc.size = nativeDevice->GetResourceAllocationInfo(0, 1, &resourceDesc).SizeInBytes;
	desc.flags = cudaExternalMemoryDedicated;
	if (!desc.handle.win32.handle || desc.size == UINT64_MAX)
		throw std::runtime_error("Could not query D3D12 shared allocation.");
	cudaExternalMemory_t memory{};
	CUDA_CHECK(cudaImportExternalMemory(&memory, &desc));
	return memory;
}

}
