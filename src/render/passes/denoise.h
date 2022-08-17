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
	DenoiseBackend() = default;

	void initialize();	

	void denoise(Color3f *rgb, Vector3f *normal, Color3f *albedo, Color3f *result);
	
	void resize(Vector2i size);
	
	void setHaveGeometry(bool haveGeometry);

private:
	Vector2i resolution;
	bool haveGeometryBuffer{}, initialized{};
	OptixDenoiser denoiserHandle;
	OptixDenoiserSizes memorySizes;
	CUDABuffer denoiserState, scratchBuffer, intensity;
};

class DenoisePass : public RenderPass {
public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<DenoisePass>;

	void render(CUDABuffer &frame) override;
	void renderUI() override;
	void resize(const Vector2i& size) override;

	string getName() const override { return "Denoiser"; }

private:
	bool mUseGeometry{};
	DenoiseBackend mBackend;
};

KRR_NAMESPACE_END