// https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"
#include "util/math_utils.h"

#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

#define METRIC_IN_SRGB					0
#define ERROR_EPS						1e-3f
#define CLAMP_PIXEL_ERROR				0 
#define CLAMP_PIXEL_ERROR_THRESHOLD		10.f
#define DISCARD_FIREFLIES				0
#define DISCARD_FIREFLIES_PRECENTAGE	0.0001f

KRR_NAMESPACE_BEGIN

namespace {
TypedBuffer<float> intermediateResult;

float *initialize_metric(size_t n_elements) {
	if (intermediateResult.size() < n_elements) {
		intermediateResult.resize(n_elements);
	}
	return intermediateResult.data();
}
}

KRR_CALLABLE float mse(const Color& y, const Color& ref) {
	return (y - ref).abs().pow(2).mean(); 
}

KRR_CALLABLE float mape(const Color &y, const Color &ref) { 
	return ((y - ref).abs() / (ref + ERROR_EPS)).mean();
}

KRR_CALLABLE float smape(const Color &y, const Color &ref) { 
	return ((y - ref).abs() / (ref + y + ERROR_EPS)).mean();
}

// is in fact MRSE...
KRR_CALLABLE float rel_mse(const Color &y, const Color &ref) {
	return ((y - ref) / (ref + ERROR_EPS)).square().mean();
	//Color ret{}, diff = (y - ref).abs();
	//for (int ch = 0; ch < Color::dim; ch++)
	//	ret[ch] = ref[ch] == 0.f ? 0.f : pow2(diff[ch] / (ref[ch]));
	//return ret.mean();
}

float calc_metric(const CudaRenderTarget & frame, const Color4f *reference, 
	size_t n_elements, ErrorMetric metric) {
	float *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int i) {	
		Color y = frame.read(i);
		Color ref = reference[i];
#if METRIC_IN_SRGB
		y = utils::linear2srgb(y);
		ref = utils::linear2srgb(ref);
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
	});

#if DISCARD_FIREFLIES
	thrust::sort(thrust::device, error_buffer, error_buffer + n_elements);
	n_elements = n_elements * (1.f - DISCARD_FIREFLIES_PRECENTAGE);
#endif

	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
				[] KRR_DEVICE(const float &val) -> float {
#if CLAMP_PIXEL_ERROR 
					return min(val, CLAMP_PIXEL_ERROR_THRESHOLD);
#endif
					return val;
				}, 0.f, thrust::plus<float>()) / n_elements;
}

KRR_NAMESPACE_END