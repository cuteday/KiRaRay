// https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"
#include "device/context.h"
#include "util/math_utils.h"

#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

#define METRIC_IN_SRGB					0
#define CLAMP_PIXEL_ERROR				1
#define DISCARD_FIREFLIES				1


NAMESPACE_BEGIN(krr)

KRR_DEVICE constexpr float ERROR_EPS					= 0;
KRR_DEVICE constexpr float CLAMP_PIXEL_ERROR_THRESHOLD	= 100.f;
KRR_DEVICE constexpr float DISCARD_FIREFLIES_PRECENTAGE = 0.0001;


namespace {
TypedBuffer<float> intermediateResult;

float *initialize_metric(size_t n_elements) {
	if (intermediateResult.size() < n_elements) {
		intermediateResult.resize(n_elements);
	}
	return intermediateResult.data();
}

KRR_CALLABLE float srgb2linear(float sRGBColor) {
	if (sRGBColor <= 0.04045f)
		return sRGBColor / 12.92f;
	else
		return pow((sRGBColor + 0.055f) / 1.055f, 2.4f);
}

KRR_CALLABLE RGB srgb2linear(RGB sRGBColor) {
	RGB ret{};
	for (int ch = 0; ch < RGB::dim; ch++)
		ret[ch] = srgb2linear(sRGBColor[ch]);
	return ret;
}

#define CHECK_INVALID(pix) if (pix.hasInf() || pix.hasNaN()) return 0.f;

KRR_CALLABLE float linear2srgb(float linearColor) {
	if (linearColor <= 0.0031308)
		return linearColor * 12.92f;
	else
		return 1.055f * pow(linearColor, 1.f / 2.4f) - 0.055f;
}

KRR_CALLABLE RGB linear2srgb(RGB linearColor) {
	RGB ret{};
	for (int ch = 0; ch < RGB::dim; ch++)
		ret[ch] = linear2srgb(linearColor[ch]);
	return ret;
}
}

KRR_CALLABLE float mse(const RGB& y, const RGB& ref) {
	CHECK_INVALID(ref)
	return (y - ref).abs().pow(2).mean(); 
}

KRR_CALLABLE float mape(const RGB &y, const RGB &ref) { 
	CHECK_INVALID(ref)
	return ((y - ref).abs() / (ref + ERROR_EPS)).mean();
}

KRR_CALLABLE float smape(const RGB &y, const RGB &ref) { 
	CHECK_INVALID(ref)
	return ((y - ref).abs() / (ref + y + ERROR_EPS)).mean();
}

// is in fact MRSE...
KRR_CALLABLE float rel_mse(const RGB &y, const RGB &ref) {
	CHECK_INVALID(ref)
	if constexpr (ERROR_EPS)
		return ((y - ref) / (ref + ERROR_EPS)).square().mean();
	else {
		RGB ret{}, diff = (y - ref).abs();
		for (int ch = 0; ch < RGB::dim; ch++)
			ret[ch] = ref[ch] == 0.f ? 0.f : pow2(diff[ch] / ref[ch]);
		return ret.mean();
	}
}

float calc_metric(CudaRenderTarget &frame, const RGBA *reference, size_t n_elements,
				  ErrorMetric metric, const bool showPixelError, const bool jetOn,
				  const float jetVMax) {
	float *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int i) {
		RGB y = frame.read(i);
		RGB ref = reference[i];
#if METRIC_IN_SRGB
		y = linear2srgb(y);
		ref = linear2srgb(ref);
#endif
		float error;
		switch (metric) {
			case ErrorMetric::MSE:
				error = mse(y, ref);
				break;
			case ErrorMetric::MAPE:
				error = mape(y, ref);
				break;
			case ErrorMetric::SMAPE:
				error = smape(y, ref);
				break;
			case ErrorMetric::RelMSE:
			default:
				error = rel_mse(y, ref);
		}
		error_buffer[i] = error;
	}, KRR_DEFAULT_STREAM);

	if (showPixelError) {
		writeErrorToRenderTarget(frame, error_buffer, n_elements, jetOn, jetVMax);
	}

#if DISCARD_FIREFLIES
	thrust::sort(thrust::device.on(KRR_DEFAULT_STREAM), error_buffer,
				 error_buffer + n_elements);
	n_elements = n_elements * (1.f - DISCARD_FIREFLIES_PRECENTAGE);
#endif

	return thrust::transform_reduce(
			   thrust::device.on(KRR_DEFAULT_STREAM), 
				error_buffer, error_buffer + n_elements,
				[] KRR_HOST_DEVICE(const float &val) -> float {
#if CLAMP_PIXEL_ERROR 
					return min(val, CLAMP_PIXEL_ERROR_THRESHOLD);
#endif
					return val;
				}, 0.f, thrust::plus<float>()) / n_elements;
}


void writeErrorToRenderTarget(CudaRenderTarget &renderTarget, float* errorBuffer, size_t n_elements, const bool jetOn,
							  const float jetVMax) {
	GPUParallelFor(
		n_elements,
		[=] KRR_DEVICE(int pixelId) mutable {
			float v = errorBuffer[pixelId];
			RGB c;
			if (jetOn) {
				// vMin = 0.f
				// https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
				c = RGB(1.f, 1.f, 1.f);
				if (v < (0.25 * jetVMax)) {
					c[0] = 0;
					c[1] = 4 * v / jetVMax;
				} else if (v < (0.5 * jetVMax)) {
					c[0] = 0;
					c[2] = 1 + 4 * (0.25 * jetVMax - v) / jetVMax;
				} else if (v < (0.75 * jetVMax)) {
					c[0] = 4 * (v - 0.5 * jetVMax) / jetVMax;
					c[2] = 0;
				} else {
					c[1] = 1 + 4 * (0.75 * jetVMax - v) / jetVMax;
					c[2] = 0;
				}
			} else {
				c = RGB(v, v, v);
			}
			renderTarget.write(RGBA(c, 1), pixelId);
		},
		KRR_DEFAULT_STREAM);
}
NAMESPACE_END(krr)