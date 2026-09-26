#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <nvrhi/d3d12.h>
#include <directx/d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cstring>

#include "graphics/device_backend.h"
#include "graphics/interop.h"
#include "logger.h"
#include "util/check.h"

NAMESPACE_BEGIN(krr)

namespace {

using Microsoft::WRL::ComPtr;

void checkD3D12(HRESULT result, const char *operation) {
	if (FAILED(result))
		throw std::runtime_error(std::string(operation) + ": HRESULT " + std::to_string(uint32_t(result)));
}

class D3D12Backend : public GraphicsBackend {
public:
	~D3D12Backend() override {
		try {
			waitForFrame(mFrameValue);
			if (mNvrhiDevice) mNvrhiDevice->waitForIdle();
		}
		catch (...) {}
		mImages.clear();
		mNvrhiDevice = nullptr;
		if (mFrameEvent) CloseHandle(mFrameEvent);
		if (mSwapChain) mSwapChain->SetFullscreenState(FALSE, nullptr);
		if (mInfoQueue) {
			for (UINT64 index = 0; index < mInfoQueue->GetNumStoredMessages(); ++index) {
				SIZE_T bytes = 0;
				mInfoQueue->GetMessage(index, nullptr, &bytes);
				std::vector<char> storage(bytes);
				auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
				if (FAILED(mInfoQueue->GetMessage(index, message, &bytes))) continue;
				if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
					std::fprintf(stderr, "D3D12: %s\n", message->pDescription);
			}
		}
	}

	void initialize(DeviceCreationParameters &params, GLFWwindow *window,
					nvrhi::IMessageCallback *callback) override;
	nvrhi::IDevice *getDevice() const override { return mNvrhiDevice; }
	const std::string &getRendererString() const override { return mRendererString; }
	std::unique_ptr<GraphicsInterop> createInterop(nvrhi::IDevice *device) override {
		return createD3D12Interop(device);
	}
	void resizeSwapChain(const DeviceCreationParameters &params) override;
	bool beginFrame() override {
		if (!mSwapChain) return true;
		mImageIndex = mSwapChain->GetCurrentBackBufferIndex();
		waitForFrame(mFrameValues[mImageIndex]);
		return true;
	}
	void present() override {
		if (!mSwapChain) return;
		const UINT flags = mTearingSupported && !mParams.vsyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
		checkD3D12(mSwapChain->Present(mParams.vsyncEnabled ? 1 : 0, flags), "Present D3D12 swapchain image");
		// DXGI presentation follows NVRHI's last command submission.
		checkD3D12(mGraphicsQueue->Signal(mFrameFence.Get(), mFrameValue + 1), "Signal D3D12 presentation fence");
		mFrameValues[mImageIndex] = ++mFrameValue;
	}
	nvrhi::ITexture *getBackBuffer(size_t index) const override {
		return index < mImages.size() ? mImages[index].Get() : nullptr;
	}
	size_t getBackBufferCount() const override { return mImages.size(); }
	size_t getCurrentBackBufferIndex() const override { return mImageIndex; }

private:
	void waitForFrame(uint64_t value) {
		if (!value || !mFrameFence) return;
		const uint64_t completed = mFrameFence->GetCompletedValue();
		if (completed == UINT64_MAX) checkD3D12(mDevice->GetDeviceRemovedReason(), "Wait for D3D12 presentation");
		if (completed >= value) return;
		checkD3D12(mFrameFence->SetEventOnCompletion(value, mFrameEvent), "Wait for D3D12 presentation");
		if (WaitForSingleObject(mFrameEvent, INFINITE) != WAIT_OBJECT_0)
			throw std::runtime_error("Could not wait for the D3D12 presentation fence.");
		checkD3D12(mDevice->GetDeviceRemovedReason(), "Complete D3D12 presentation");
	}

	DeviceCreationParameters mParams;
	HWND mWindow = nullptr;
	ComPtr<IDXGIFactory4> mFactory;
	ComPtr<ID3D12Device> mDevice;
	ComPtr<ID3D12InfoQueue> mInfoQueue;
	ComPtr<ID3D12CommandQueue> mGraphicsQueue;
	ComPtr<ID3D12CommandQueue> mComputeQueue;
	ComPtr<ID3D12CommandQueue> mCopyQueue;
	ComPtr<IDXGISwapChain3> mSwapChain;
	nvrhi::DeviceHandle mNvrhiDevice;
	std::vector<nvrhi::TextureHandle> mImages;
	ComPtr<ID3D12Fence> mFrameFence;
	HANDLE mFrameEvent = nullptr;
	uint64_t mFrameValue = 0;
	std::vector<uint64_t> mFrameValues;
	uint32_t mImageIndex = 0;
	bool mTearingSupported = false;
	std::string mRendererString;
};

void D3D12Backend::initialize(DeviceCreationParameters &params, GLFWwindow *window,
							  nvrhi::IMessageCallback *callback) {
	mParams = params;
	mWindow = window ? glfwGetWin32Window(window) : nullptr;
	bool debugEnabled = false;
	if (params.enableDebugRuntime) {
		ComPtr<ID3D12Debug> debug;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
			debug->EnableDebugLayer();
			debugEnabled = true;
		} else Log(Warning, "D3D12 debug layer is unavailable.");
	}
	checkD3D12(CreateDXGIFactory2(debugEnabled ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&mFactory)), "Create DXGI factory");
	int cudaDevice = 0;
	cudaDeviceProp cudaProperties{};
	CUDA_CHECK(cudaGetDevice(&cudaDevice));
	CUDA_CHECK(cudaGetDeviceProperties(&cudaProperties, cudaDevice));
	ComPtr<IDXGIAdapter1> selected;
	for (UINT index = 0;; ++index) {
		ComPtr<IDXGIAdapter1> adapter;
		const HRESULT result = mFactory->EnumAdapters1(index, &adapter);
		if (result == DXGI_ERROR_NOT_FOUND) break;
		checkD3D12(result, "Enumerate DXGI adapters");
		DXGI_ADAPTER_DESC1 properties{};
		checkD3D12(adapter->GetDesc1(&properties), "Query DXGI adapter");
		if (properties.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
		if (std::memcmp(cudaProperties.luid, &properties.AdapterLuid, sizeof(LUID)) == 0) {
			selected = adapter;
			break;
		}
	}
	if (!selected) throw std::runtime_error("No D3D12 adapter matches the active CUDA device.");
	checkD3D12(D3D12CreateDevice(selected.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&mDevice)), "Create D3D12 device");
	if (mDevice->GetNodeCount() != 1) throw std::runtime_error("CUDA interop requires a single-node D3D12 device.");
	if (debugEnabled) mDevice.As(&mInfoQueue);
	mRendererString = cudaProperties.name;
	auto createQueue = [this](D3D12_COMMAND_LIST_TYPE type, ComPtr<ID3D12CommandQueue> &queue) {
		D3D12_COMMAND_QUEUE_DESC desc{};
		desc.Type = type;
		checkD3D12(mDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "Create D3D12 command queue");
	};
	createQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, mGraphicsQueue);
	if (params.enableComputeQueue) createQueue(D3D12_COMMAND_LIST_TYPE_COMPUTE, mComputeQueue);
	if (params.enableCopyQueue) createQueue(D3D12_COMMAND_LIST_TYPE_COPY, mCopyQueue);
	nvrhi::d3d12::DeviceDesc desc{};
	desc.errorCB = callback;
	desc.pDevice = mDevice.Get();
	desc.pGraphicsCommandQueue = mGraphicsQueue.Get();
	desc.pComputeCommandQueue = mComputeQueue.Get();
	desc.pCopyCommandQueue = mCopyQueue.Get();
	mNvrhiDevice = nvrhi::d3d12::createDevice(desc);
	if (!mNvrhiDevice) throw std::runtime_error("NVRHI could not create its D3D12 device.");
	ComPtr<IDXGIFactory5> factory5;
	if (SUCCEEDED(mFactory.As(&factory5))) {
		BOOL supported = FALSE;
		if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &supported, sizeof(supported))))
			mTearingSupported = supported != FALSE;
	}
	if (mWindow) resizeSwapChain(params);
	Log(Success, "Created D3D12 device: %s", mRendererString.c_str());
}

void D3D12Backend::resizeSwapChain(const DeviceCreationParameters &params) {
	if (!mWindow) return;
	waitForFrame(mFrameValue);
	mNvrhiDevice->waitForIdle();
	mNvrhiDevice->runGarbageCollection();
	mImages.clear();
	mParams = params;
	DXGI_FORMAT format = nvrhi::d3d12::convertFormat(params.swapChainFormat);
	if (format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) format = DXGI_FORMAT_R8G8B8A8_UNORM;
	if (format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) format = DXGI_FORMAT_B8G8R8A8_UNORM;
	const UINT flags = mTearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	const UINT count = std::max(2u, params.swapChainBufferCount);
	if (mSwapChain) {
		checkD3D12(mSwapChain->ResizeBuffers(count, params.backBufferWidth, params.backBufferHeight, format, flags), "Resize D3D12 swapchain");
	} else {
		DXGI_SWAP_CHAIN_DESC1 desc{};
		desc.Width = params.backBufferWidth;
		desc.Height = params.backBufferHeight;
		desc.Format = format;
		desc.SampleDesc.Count = 1;
		desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		desc.BufferCount = count;
		desc.Scaling = DXGI_SCALING_STRETCH;
		desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
		desc.Flags = flags;
		ComPtr<IDXGISwapChain1> swapChain;
		checkD3D12(mFactory->CreateSwapChainForHwnd(mGraphicsQueue.Get(), mWindow, &desc, nullptr, nullptr, &swapChain), "Create D3D12 swapchain");
		checkD3D12(swapChain.As(&mSwapChain), "Query D3D12 swapchain");
		checkD3D12(mFactory->MakeWindowAssociation(mWindow, DXGI_MWA_NO_ALT_ENTER), "Set DXGI window association");
		checkD3D12(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFrameFence)), "Create D3D12 presentation fence");
		mFrameEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		if (!mFrameEvent) throw std::runtime_error("Could not create the D3D12 presentation event.");
	}
	for (UINT index = 0; index < count; ++index) {
		ComPtr<ID3D12Resource> resource;
		checkD3D12(mSwapChain->GetBuffer(index, IID_PPV_ARGS(&resource)), "Get D3D12 swapchain image");
		nvrhi::TextureDesc desc;
		desc.width = params.backBufferWidth;
		desc.height = params.backBufferHeight;
		desc.format = params.swapChainFormat;
		desc.isRenderTarget = true;
		desc.initialState = nvrhi::ResourceStates::Present;
		desc.keepInitialState = true;
		desc.debugName = "Swapchain image";
		mImages.push_back(mNvrhiDevice->createHandleForNativeTexture(nvrhi::ObjectTypes::D3D12_Resource, resource.Get(), desc));
	}
	mFrameValues.assign(count, 0);
	mImageIndex = mSwapChain->GetCurrentBackBufferIndex();
}

} // namespace

std::unique_ptr<GraphicsBackend> createD3D12Backend() { return std::make_unique<D3D12Backend>(); }

NAMESPACE_END(krr)
