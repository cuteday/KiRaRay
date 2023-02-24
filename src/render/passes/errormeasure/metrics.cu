// https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"
#include "util/math_utils.h"

#include "thrust/reduce.h"
#include "thrust/transform_reduce.h"
#include "thrust/execution_policy.h"

#define METRIC_IN_SRGB	0
#define ERROR_EPS 1e-2f

KRR_NAMESPACE_BEGIN

namespace {
TypedBuffer<Color3f> intermediateResult;

Color3f *initialize_metric(size_t n_elements) {
	if (intermediateResult.size() < n_elements) {
		intermediateResult.resize(n_elements);
	}
	return intermediateResult.data();
}
}

KRR_CALLABLE Color mse(const Color& y, const Color& ref) {
	return (y - ref).abs().pow(2); 
}

KRR_CALLABLE Color mape(const Color &y, const Color &ref) { 
	return (y - ref).abs() / (ref + ERROR_EPS);
}

KRR_CALLABLE Color smape(const Color &y, const Color &ref) { 
	return (y - ref).abs() / (ref + y + ERROR_EPS);
}

// is in fact MRSE...
KRR_CALLABLE Color rel_mse(const Color &y, const Color &ref) {
	Color ret{}, diff = (y - ref).abs();
	for (int ch = 0; ch < Color::dim; ch++) {
		ret[ch] = ref[ch] == 0.f ? 0.f : pow2(diff[ch] / ref[ch]);
	}
	return ret;
}

float calc_metric(const CudaRenderTarget & frame, const Color4f *reference, 
	size_t n_elements, ErrorMetric metric) {
	Color3f *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int i) {	
		Color y = frame.read(i);
		Color ref = reference[i];
#if METRIC_IN_SRGB
		y = utils::linear2srgb(y);
		ref = utils::linear2srgb(ref);
#endif
		Color error;
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
	
	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
				[] KRR_DEVICE(const Color3f &val) -> float {
					return val.mean(); }, 0.f, thrust::plus<float>()) / n_elements;
}

KRR_NAMESPACE_END