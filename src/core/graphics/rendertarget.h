#pragma once

#include "common.h"
#include "device/cuda.h"
#include "interop.h"

namespace krr {

class RenderTexture {
public:
	using SharedPtr = std::shared_ptr<RenderTexture>;
	RenderTexture(nvrhi::IDevice *device, const nvrhi::TextureDesc &desc);
	static SharedPtr create(nvrhi::IDevice *device, Vector2i size, nvrhi::Format format,
		const std::string &name = "");

	operator nvrhi::TextureHandle() const { return mMapping.getTexture(); }
	operator nvrhi::ITexture *() const { return mMapping.getTexture(); }
	operator CudaRenderTarget() const { return getCudaRenderTarget(); }
	nvrhi::ITexture *getTexture() const { return mMapping.getTexture(); }
	Vector2i getSize() const;
	CudaRenderTarget getCudaRenderTarget() const;

private:
	CudaTextureMapping mMapping;
};

class RenderTarget {
public:
	using SharedPtr = std::shared_ptr<RenderTarget>;
	explicit RenderTarget(nvrhi::IDevice *device, CUstream stream = nullptr) :
		mDevice(device), mCudaStream(stream) {}

	nvrhi::IFramebuffer *getFramebuffer() const { return mFramebuffer.Get(); }
	RenderTexture *getColorTexture() const { return mColor.get(); }
	RenderTexture *getDepthTexture() const { return mDepth.get(); }
	RenderTexture *getDiffuseTexture() const { return mDiffuse.get(); }
	RenderTexture *getSpecularTexture() const { return mSpecular.get(); }
	RenderTexture *getNormalTexture() const { return mNormal.get(); }
	RenderTexture *getEmissiveTexture() const { return mEmissive.get(); }
	RenderTexture *getMotionTexture() const { return mMotion.get(); }
	std::vector<nvrhi::ITexture *> getTextures() const;

	void setDepthEnabled(bool enable) { setEnabled(mEnableDepth, enable); }
	void setDiffuseEnabled(bool enable) { setEnabled(mEnableDiffuse, enable); }
	void setSpecularEnabled(bool enable) { setEnabled(mEnableSpecular, enable); }
	void setNormalEnabled(bool enable) { setEnabled(mEnableNormal, enable); }
	void setEmissiveEnabled(bool enable) { setEnabled(mEnableEmissive, enable); }
	void setMotionEnabled(bool enable) { setEnabled(mEnableMotion, enable); }

	void resize(Vector2i size);
	Vector2i getSize() const { return mSize; }
	bool isUpdateNeeded(Vector2i size) const { return size != mSize; }

private:
	friend class RenderContext;
	void setEnabled(bool &enabled, bool value);
	bool mCudaActive{};
	nvrhi::IDevice *mDevice{};
	CUstream mCudaStream{};
	RenderTexture::SharedPtr mColor;
	RenderTexture::SharedPtr mDepth;
	RenderTexture::SharedPtr mDiffuse;
	RenderTexture::SharedPtr mSpecular;
	RenderTexture::SharedPtr mNormal;
	RenderTexture::SharedPtr mEmissive;
	RenderTexture::SharedPtr mMotion;
	nvrhi::FramebufferHandle mFramebuffer;
	bool mEnableDepth{};
	bool mEnableDiffuse{};
	bool mEnableSpecular{};
	bool mEnableNormal{};
	bool mEnableEmissive{};
	bool mEnableMotion{};
	Vector2i mSize{};
};

}
