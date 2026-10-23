#pragma once

#include "rendertarget.h"
#include <scene.h>
#include <array>

namespace krr {

class RenderContext {
public:
	using SharedPtr = std::shared_ptr<RenderContext>;
	struct Readback;
	using ReadbackHandle = std::shared_ptr<Readback>;
	struct CudaScope {
		explicit CudaScope(RenderContext *context);
		~CudaScope() noexcept(false);
		CudaScope(const CudaScope &) = delete;
		CudaScope &operator=(const CudaScope &) = delete;
	private:
		RenderContext *mContext;
		bool mEntered;
	};

	RenderContext(nvrhi::IDevice *device, std::unique_ptr<GraphicsInterop> interop);
	~RenderContext();
	RenderContext(const RenderContext &) = delete;
	RenderContext &operator=(const RenderContext &) = delete;

	nvrhi::IDevice *getDevice() const { return mDevice; }
	nvrhi::ICommandList *getCommandList() const { return mCommandList.Get(); }
	nvrhi::IFramebuffer *getFramebuffer() const { return mRenderTarget->getFramebuffer(); }
	RenderTexture *getColorTexture() const { return mRenderTarget->getColorTexture(); }
	CUstream getCudaStream() const { return mCudaStream; }
	RenderTarget::SharedPtr getRenderTarget() const { return mRenderTarget; }
	Scene::SharedPtr getScene() const { return mScene; }

	void setScene(Scene::SharedPtr scene) { mScene = std::move(scene); }
	void resize(Vector2i size);
	void beginCuda();
	void endCuda();
	// The mapping must outlive its registration.
	void addSharedBuffer(nvrhi::IBuffer *buffer);
	void removeSharedBuffer(nvrhi::IBuffer *buffer);
	void clear();
	std::vector<float> readback();
	ReadbackHandle enqueueReadback();
	bool isReadbackReady(const ReadbackHandle &readback);
	std::vector<float> collectReadback(const ReadbackHandle &readback, bool wait = false);

private:
	std::vector<nvrhi::IBuffer *> getSharedBuffers() const;
	std::vector<float> copyReadback(nvrhi::IStagingTexture *staging);
	void releaseReadbacks();
	nvrhi::DeviceHandle mDevice;
	std::unique_ptr<GraphicsInterop> mInterop;
	Scene::SharedPtr mScene;
	nvrhi::CommandListHandle mCommandList;
	nvrhi::StagingTextureHandle mReadback;
	std::array<ReadbackHandle, 2> mAsyncReadbacks;
	RenderTarget::SharedPtr mRenderTarget;
	std::vector<nvrhi::BufferHandle> mSharedBuffers;
	std::vector<nvrhi::ITexture *> mCudaTextures;
	CUstream mCudaStream{};
	bool mCudaActive{};
};

}
