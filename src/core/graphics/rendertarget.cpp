#include "rendertarget.h"

namespace krr {

RenderTexture::RenderTexture(nvrhi::IDevice *device, const nvrhi::TextureDesc &desc) :
	mMapping(device, desc) {}

RenderTexture::SharedPtr RenderTexture::create(nvrhi::IDevice *device, Vector2i size,
	nvrhi::Format format, const std::string &name) {
	nvrhi::TextureDesc desc;
	desc.width = size[0];
	desc.height = size[1];
	desc.format = format;
	desc.debugName = name;
	desc.isRenderTarget = true;
	desc.setClearValue(nvrhi::Color(0.f));
	return std::make_shared<RenderTexture>(device, desc);
}

Vector2i RenderTexture::getSize() const {
	const auto &desc = getTexture()->getDesc();
	return Vector2i{desc.width, desc.height};
}

CudaRenderTarget RenderTexture::getCudaRenderTarget() const {
	const auto &desc = getTexture()->getDesc();
	int channels = nvrhi::getFormatInfo(desc.format).bytesPerBlock / sizeof(float);
	return CudaRenderTarget{mMapping.getSurface(), int(desc.width), int(desc.height), channels};
}

void RenderTarget::resize(Vector2i size) {
	if (size[0] < 0 || size[1] < 0) throw std::invalid_argument("Render target dimensions must be nonnegative.");
	if (mCudaActive) throw std::logic_error("Cannot resize a render target during CUDA access.");
	CUDA_CHECK(cudaStreamSynchronize(mCudaStream));
	if (!mDevice->waitForIdle()) throw std::runtime_error("Graphics device lost while resizing render targets.");
	mSize = size;
	mFramebuffer = nullptr;
	if (size[0] == 0 || size[1] == 0) {
		mColor.reset();
		mDepth.reset();
		mDiffuse.reset();
		mSpecular.reset();
		mNormal.reset();
		mEmissive.reset();
		mMotion.reset();
		return;
	}
	mColor = RenderTexture::create(mDevice, size, nvrhi::Format::RGBA32_FLOAT, "Color Texture");
	auto create = [&](bool enabled, nvrhi::Format format, const char *name) {
		return enabled ? RenderTexture::create(mDevice, size, format, name) : nullptr;
	};
	mDepth = create(mEnableDepth, nvrhi::Format::R32_FLOAT, "Depth Texture");
	mDiffuse = create(mEnableDiffuse, nvrhi::Format::RGBA32_FLOAT, "Diffuse Texture");
	mSpecular = create(mEnableSpecular, nvrhi::Format::RGBA32_FLOAT, "Specular Texture");
	mNormal = create(mEnableNormal, nvrhi::Format::RGBA32_FLOAT, "Normal Texture");
	mMotion = create(mEnableMotion, nvrhi::Format::RGBA32_FLOAT, "Motion Texture");
	mEmissive = create(mEnableEmissive, nvrhi::Format::RGBA32_FLOAT, "Emissive Texture");
	mFramebuffer = mDevice->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(mColor->getTexture()));
	if (!mFramebuffer) throw std::runtime_error("Could not create the render framebuffer.");
}

void RenderTarget::setEnabled(bool &enabled, bool value) {
	if (enabled == value) return;
	if (mCudaActive) throw std::logic_error("Cannot replace render textures during CUDA access.");
	enabled = value;
	resize(mSize);
}

std::vector<nvrhi::ITexture *> RenderTarget::getTextures() const {
	std::vector<nvrhi::ITexture *> textures;
	for (const auto *texture : {mColor.get(), mDepth.get(), mDiffuse.get(), mSpecular.get(),
		mNormal.get(), mEmissive.get(), mMotion.get()})
		if (texture) textures.push_back(texture->getTexture());
	return textures;
}

}
