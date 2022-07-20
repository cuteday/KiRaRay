#include "tonemapping.h"

#include "device/cuda.h"
#include "device/context.h"
#include "math/utils.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
KRR_CALLABLE Color toneMapAces(Color color) {
		color *= 0.6;
		float A = 2.51;
		float B = 0.03;
		float C = 2.43;
		float D = 0.59;
		float E = 0.14;
		color	= clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0, 1);
		return color;
	}

	KRR_CALLABLE Color toneMapReinhard(Color color) {
		float luminance = utils::luminance(color);
		float reinhard = luminance / (luminance + 1);
		return color * (reinhard / luminance);
	}

	KRR_CALLABLE Color toneMapUC2(Color color) {
		float A = 0.22; // Shoulder Strength
		float B = 0.3;  // Linear Strength
		float C = 0.1;  // Linear Angle
		float D = 0.2;  // Toe Strength
		float E = 0.01; // Toe Numerator
		float F = 0.3;  // Toe Denominator
		color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - (E / F);
		return color;
	}

	KRR_CALLABLE Color toneMapHejiHableAlu(Color color) {
		color = (color - 0.004f).cwiseMax(0);
		color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
		// Result includes sRGB conversion
		return pow(color, Color(2.2));
	}
}

void ToneMappingPass::renderUI() {
	static const char *operators[] = { "Linear", "Reinhard", "Aces", "Uncharted2", "HejiHable" };
	if (ui::CollapsingHeader("Tone mapping pass")) {
		ui::Checkbox("Enabled", &mEnable);
		if (mEnable) {
			ui::DragFloat("Exposure compensation", &mExposureCompensation, 0.001, 0.001, 100, "%.3f");
			ui::Combo("Tonemap operator", (int *) &mOperator, operators, (int) Operator::NumsOperators);
		}
	}
}

void ToneMappingPass::render(CUDABuffer& frame){
	if (!mEnable) return;
	PROFILE("Tong mapping pass");
	CUstream &stream = gpContext->cudaStream;
	Color colorTransform = Color(mExposureCompensation);
	Color4f *frameBuffer = (Color4f *) frame.data();
	GPUParallelFor(mFrameSize[0] * mFrameSize[1], KRR_DEVICE_LAMBDA(int pixelId) {
		Color color = Color(frameBuffer[pixelId]) * colorTransform;
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
		frameBuffer[pixelId] = Color4f(color, 1.f);
	});
}

KRR_NAMESPACE_END