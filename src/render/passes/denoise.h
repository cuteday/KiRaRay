#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "device/buffer.h"
#include "math/math.h"
#include "renderpass.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

class DenoiseBackend {
public:
	enum class PixelFormat {
		FLOAT3,
		FLOAT4
	};

	DenoiseBackend() = default;

	void initialize();	

	void denoise(float *rgb, float *normal, float *albedo, float *result);
	
	void resize(Vector2i size);
	
	void setHaveGeometry(bool haveGeometry);

	void setPixelFormat(PixelFormat format);

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
	KRR_CLASS_DEFINE(DenoisePass, mUseGeometry);

	void render(CUDABuffer &frame) override;
	void renderUI() override;
	void resize(const Vector2i& size) override;

	string getName() const override { return "DenoisePass"; }

private:
	bool mUseGeometry{};
	DenoiseBackend mBackend;
};

KRR_NAMESPACE_END