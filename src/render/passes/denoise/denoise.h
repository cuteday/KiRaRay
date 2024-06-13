#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "device/buffer.h"

#include "renderpass.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class DenoiseBackend {
public:
	enum class PixelFormat {
		FLOAT3,
		FLOAT4
	};

	DenoiseBackend() = default;

	void initialize();

	void denoise(CUstream stream, float *rgb, float *normal, float *albedo, float *result);

	void resize(Vector2i size);

	void setHaveGeometry(bool haveGeometry);

	void setPixelFormat(PixelFormat format);
	void setProps(bool haveGeometry, PixelFormat format);

private:
	Vector2i resolution;
	PixelFormat pixelFormat{PixelFormat::FLOAT4};
	bool haveGeometryBuffer{}, initialized{};
	OptixDenoiser denoiserHandle{};
	OptixDenoiserSizes memorySizes;
	CUDABuffer denoiserState, scratchBuffer, intensity;
};

class DenoisePass : public RenderPass {
public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<DenoisePass>;
	KRR_REGISTER_PASS_DEC(DenoisePass);

	void render(RenderContext *context) override;
	void renderUI() override;
	void resize(const Vector2i &size) override;
	// true: have special task(CtxDenoiseGBuffer)
	void checkSpecTask(RenderContext *context);
	string getName() const override { return "DenoisePass"; }

	friend void from_json(const json &j, DenoisePass &p) {
		p.mDoSpecTask  = j.value("doSpecTask", false);
		p.mUseGeometry = j.value("useGeometry", false);
	}

	friend void to_json(json &j, const DenoisePass &p) {
		j.update({{"doSpecTask", p.mDoSpecTask}, {"useGeometry", p.mUseGeometry}});
	}

public:
	constexpr static char CTX_JSON_GBUFFER[] = "DENOISE_GBUFFER";

private:
	bool mDoSpecTask{false};
	bool mUseGeometry{};
	TypedBuffer<RGBA> mColorBuffer;
	DenoiseBackend mBackend;
};

typedef struct {
	bool mState;
	float *mColorBuffer;
	float *mAlbedoBuffer;
	float *mNormalBuffer;
	float *mDenoisedBuffer;
	DenoiseBackend::PixelFormat mPixelFormat;
} CtxDenoiseGBuffer;

NAMESPACE_END(krr)