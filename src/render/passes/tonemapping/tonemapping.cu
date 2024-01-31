#include "tonemapping.h"

#include "device/cuda.h"
#include "device/context.h"
#include "util/math_utils.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

namespace {
KRR_CALLABLE RGB toneMapAces(RGB color) {
		color *= 0.6;
		float A = 2.51;
		float B = 0.03;
		float C = 2.43;
		float D = 0.59;
		float E = 0.14;
		color	= clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0, 1);
		return color;
	}

	KRR_CALLABLE RGB toneMapReinhard(RGB color) {
		float lum = luminance(color);
		float reinhard = lum / (lum + 1);
		return RGB(color * reinhard).safeDiv(lum);
	}

	KRR_CALLABLE RGB toneMapUC2(RGB color) {
		float A = 0.22; // Shoulder Strength
		float B = 0.3;  // Linear Strength
		float C = 0.1;  // Linear Angle
		float D = 0.2;  // Toe Strength
		float E = 0.01; // Toe Numerator
		float F = 0.3;  // Toe Denominator
		color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - (E / F);
		return color;
	}

	KRR_CALLABLE RGB toneMapHejiHableAlu(RGB color) {
		color = (color - 0.004f).cwiseMax(0);
		color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
		// Result includes sRGB conversion
		return color.pow(2.2);
	}
}

void ToneMappingPass::renderUI() {
	static const char *operators[] = { "Linear", "Reinhard", "Aces", "Uncharted2", "HejiHable" };
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		ui::DragFloat("Exposure compensation", &mExposureCompensation, 0.01, 0.01, 100, "%.2f");
		ui::Combo("Tonemap operator", (int *) &mOperator, operators, (int) Operator::NumsOperators);
		ui::Checkbox("Use Gamma", &mUseGamma);
	}
}

void ToneMappingPass::render(RenderContext *context) {
	PROFILE("Tong mapping pass");
	CUstream &stream = gpContext->cudaStream;
	RGB colorTransform = RGB(mExposureCompensation);
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();
	GPUParallelFor(getFrameSize()[0] * getFrameSize()[1], KRR_DEVICE_LAMBDA(int pixelId) {
		RGB color = frameBuffer.read(pixelId).head<3>() * colorTransform;
		switch (mOperator) {
		case krr::ToneMappingPass::Operator::Linear:
			break;
		case krr::ToneMappingPass::Operator::Reinhard:
			color = toneMapReinhard(color);
			break;
		case krr::ToneMappingPass::Operator::Aces:
			color = toneMapAces(color);
			break;
		case krr::ToneMappingPass::Operator::Uncharted2:
			color = toneMapUC2(color);
			break;
		case krr::ToneMappingPass::Operator::HejiHable:
			color = toneMapHejiHableAlu(color);
			break;
		default:
			break;
		}
		if (mUseGamma) color = color.pow(0.45454545f);
		frameBuffer.write(RGBA(color, 1.f), pixelId);
	});
}

KRR_REGISTER_PASS_DEF(ToneMappingPass);
NAMESPACE_END(krr)