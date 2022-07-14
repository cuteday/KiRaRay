#include "tonemapping.h"

#include "device/cuda.h"
#include "device/context.h"
#include "math/utils.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace {
	KRR_CALLABLE vec3f toneMapAces(vec3f color) {
		color *= 0.6;
		float A = 2.51;
		float B = 0.03;
		float C = 2.43;
		float D = 0.59;
		float E = 0.14;
		color = saturate((color * (A * color + B)) / (color * (C * color + D) + E));
		return color;
	}

	KRR_CALLABLE vec3f toneMapReinhard(vec3f color) {
		float luminance = utils::luminance(color);
		float reinhard = luminance / (luminance + 1);
		return color * (reinhard / luminance);
	}

	KRR_CALLABLE vec3f toneMapUC2(vec3f color) {
		float A = 0.22; // Shoulder Strength
		float B = 0.3;  // Linear Strength
		float C = 0.1;  // Linear Angle
		float D = 0.2;  // Toe Strength
		float E = 0.01; // Toe Numerator
		float F = 0.3;  // Toe Denominator
		color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - (E / F);
		return color;
	}

	KRR_CALLABLE vec3f toneMapHejiHableAlu(vec3f color) {
		color = max(vec3f(0), color - 0.004f);
		color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
		// Result includes sRGB conversion
		return pow(color, vec3f(2.2));
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
	vec3f colorTransform = vec3f(mExposureCompensation);
	vec4f* frameBuffer = (vec4f*)frame.data();
	GPUParallelFor(mFrameSize.x * mFrameSize.y, KRR_DEVICE_LAMBDA(int pixelId) {
		vec3f color = vec3f(frameBuffer[pixelId]) * colorTransform;
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
		frameBuffer[pixelId] = vec4f(color, 1.f);
	});
}

KRR_NAMESPACE_END