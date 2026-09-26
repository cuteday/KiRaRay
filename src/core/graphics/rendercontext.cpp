#include "rendercontext.h"
#include "device/context.h"
#include <exception>

namespace krr {

RenderContext::RenderContext(nvrhi::IDevice *device, std::unique_ptr<GraphicsInterop> interop) :
	mDevice(device), mInterop(std::move(interop)), mCudaStream(KRR_DEFAULT_STREAM) {
	if (!mInterop) throw std::invalid_argument("Graphics interop is required.");
	mRenderTarget = std::make_shared<RenderTarget>(device, mCudaStream);
	mCommandList = mDevice->createCommandList();
	if (!mCommandList) throw std::runtime_error("Could not create a render command list.");
}

RenderContext::~RenderContext() {
	try { endCuda(); }
	catch (...) { std::fprintf(stderr, "Could not complete the CUDA handoff during cleanup.\n"); }
	auto result = cudaStreamSynchronize(mCudaStream);
	if (result != cudaSuccess)
		std::fprintf(stderr, "CUDA cleanup: %s\n", cudaGetErrorString(result));
	try {
		if (!mDevice->waitForIdle()) std::fprintf(stderr, "Graphics device lost during cleanup.\n");
	} catch (...) {
		std::fprintf(stderr, "Could not wait for graphics work during cleanup.\n");
	}
	mCommandList = nullptr;
	mRenderTarget.reset();
	mInterop.reset();
}

RenderContext::CudaScope::CudaScope(RenderContext *context) :
	mContext(context), mEntered(!context->mCudaActive) {
	mContext->beginCuda();
}

RenderContext::CudaScope::~CudaScope() noexcept(false) {
	if (!mEntered) return;
	if (std::uncaught_exceptions()) {
		try { mContext->endCuda(); } catch (...) {}
	} else mContext->endCuda();
}

void RenderContext::beginCuda() {
	if (mCudaActive) return;
	mCudaTextures = mRenderTarget->getTextures();
	mInterop->beginCuda(mCudaTextures, getSharedBuffers(), mCudaStream);
	mCudaActive = true;
	mRenderTarget->mCudaActive = true;
}

void RenderContext::endCuda() {
	if (!mCudaActive) return;
	mInterop->endCuda(mCudaTextures, getSharedBuffers(), mCudaStream);
	mCudaActive = false;
	mRenderTarget->mCudaActive = false;
	mCudaTextures.clear();
}

void RenderContext::addSharedBuffer(nvrhi::IBuffer *buffer) {
	if (mCudaActive) throw std::logic_error("Cannot register shared buffers during CUDA rendering.");
	if (!buffer || (buffer->getDesc().sharedResourceFlags & nvrhi::SharedResourceFlags::Shared) == 0)
		throw std::invalid_argument("A shared graphics buffer is required.");
	if (std::find(mSharedBuffers.begin(), mSharedBuffers.end(), buffer) == mSharedBuffers.end())
		mSharedBuffers.push_back(buffer);
}

void RenderContext::removeSharedBuffer(nvrhi::IBuffer *buffer) {
	endCuda();
	CUDA_CHECK(cudaStreamSynchronize(mCudaStream));
	if (!mDevice->waitForIdle()) throw std::runtime_error("Graphics device lost while unregistering a shared buffer.");
	mSharedBuffers.erase(std::remove(mSharedBuffers.begin(), mSharedBuffers.end(), buffer), mSharedBuffers.end());
}

std::vector<nvrhi::IBuffer *> RenderContext::getSharedBuffers() const {
	std::vector<nvrhi::IBuffer *> buffers;
	for (const auto &buffer : mSharedBuffers) buffers.push_back(buffer.Get());
	return buffers;
}

void RenderContext::resize(Vector2i size) {
	endCuda();
	mRenderTarget->resize(size);
}

void RenderContext::clear() {
	endCuda();
	mCommandList->open();
	for (auto *texture : mRenderTarget->getTextures())
		mCommandList->clearTextureFloat(texture, nvrhi::AllSubresources, nvrhi::Color(0.f));
	mCommandList->close();
	mDevice->executeCommandList(mCommandList);
}

std::vector<float> RenderContext::readback() {
	endCuda();
	if (!getColorTexture()) throw std::logic_error("Cannot read an empty render target.");
	auto *texture = getColorTexture()->getTexture();
	auto desc = texture->getDesc();
	desc.sharedResourceFlags = nvrhi::SharedResourceFlags::None;
	auto staging = mDevice->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);
	if (!staging) throw std::runtime_error("Could not create image readback storage.");
	mCommandList->open();
	mCommandList->copyTexture(staging, {}, texture, {});
	mCommandList->close();
	mDevice->executeCommandList(mCommandList);
	if (!mDevice->waitForIdle()) throw std::runtime_error("Graphics device lost during image readback.");
	std::vector<float> result(size_t(desc.width) * desc.height * 3);
	size_t pitch{};
	const auto *data = static_cast<const unsigned char *>(mDevice->mapStagingTexture(
		staging, {}, nvrhi::CpuAccessMode::Read, &pitch));
	if (!data) throw std::runtime_error("Could not map the rendered image.");
	for (uint32_t y = 0; y < desc.height; ++y) {
		const auto *row = reinterpret_cast<const float *>(data + y * pitch);
		for (uint32_t x = 0; x < desc.width; ++x)
			std::copy_n(row + x * 4, 3, result.data() + (size_t(y) * desc.width + x) * 3);
	}
	mDevice->unmapStagingTexture(staging);
	return result;
}

}
