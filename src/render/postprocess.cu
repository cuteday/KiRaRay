#include "postprocess.h"
#include "math/utils.h"

#include "device/context.h"

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace shader;

namespace shader {

	template<typename T>
    __global__ void accumulateFrame(uint n_elements, vec4f* currentBuffer, vec4f* accumBuffer, uint accumCount, bool average)
    {
		uint i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_elements) return;
		float currentWeight = 1.f / (accumCount + 1);
		if (accumCount > 0) {
			if(average)
				// moving average mode
				accumBuffer[i] = utils::lerp(accumBuffer[i], currentBuffer[i], currentWeight);
			else
				// sum mode
				accumBuffer[i] = accumBuffer[i] + currentBuffer[i];
		}
		else {
			accumBuffer[i] = currentBuffer[i];
		}
		if (average)
			currentBuffer[i] = accumBuffer[i];
		else
			currentBuffer[i] = accumBuffer[i] * currentWeight;
	}
}

void AccumulatePass::render(CUDABuffer& frame) {

	if (!mEnable) return;
	if (mpScene->getChanges()) reset();
	CUstream& stream = gpContext->cudaStream;
	linear_kernel(accumulateFrame<vec4f>, 0, stream, mFrameSize.x * mFrameSize.y, 
		(vec4f*)frame.data(), (vec4f*)mAccumBuffer.data(), mAccumCount, false);

	mAccumCount = min(mAccumCount + 1, mMaxAccumCount - 1);
}

namespace shader {
	namespace tonemapper {
		KRR_CALLABLE vec3f toneMapAces(vec3f color) {
			// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
			color *= 0.6;
			float A = 2.51;
			float B = 0.03;
			float C = 2.43;
			float D = 0.59;
			float E = 0.14;
			color = saturate((color * (A * color + B)) / (color * (C * color + D) + E));
			return color;
		}

		KRR_CALLABLE vec3f toneMapReinhard(vec3f color)
		{
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

		KRR_CALLABLE vec3f toneMapHejiHableAlu(vec3f color)
		{
			color = max(vec3f(0), color - 0.004f);
			color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
			// Result includes sRGB conversion
			return pow(color, vec3f(2.2));
		}
	}

	using namespace tonemapper;

	template <typename T>
	__global__ void toneMap(uint n_elements, vec4f* frame, vec3f colorTransform, ToneMappingPass::Operator toneMapOperator) {
		uint i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_elements) return;
		vec3f color = vec3f(frame[i]) * colorTransform;
		switch (toneMapOperator)
		{
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
		frame[i] = vec4f(color, 1.f);
	}
}

void ToneMappingPass::render(CUDABuffer& frame)
{
	if (!mEnable) return;
	CUstream &stream = gpContext->cudaStream;
	vec3f colorTransform = vec3f(mExposureCompensation);
	linear_kernel(toneMap<float>, 0, stream, mFrameSize.x * mFrameSize.y,
		(vec4f*)frame.data(), colorTransform, mOperator);
	
}

KRR_NAMESPACE_END


