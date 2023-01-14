// https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"
#include "util/math_utils.h"

#include "thrust/reduce.h"
#include "thrust/transform_reduce.h"
#include "thrust/execution_policy.h"

#define METRIC_IN_SRGB	0

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

float calc_metric_mse(const Color4f *frame, const Color4f *reference,  
	size_t n_elements) {
	Color3f *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable { 
		Color3f difference = (frame[pixel] - reference[pixel]).head<3>();
		error_buffer[pixel] = difference.abs();
	});
	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return val.pow(2).mean(); }, 
			0.f, thrust::plus<float>()) / n_elements;
}

float calc_metric_mape(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	Color3f *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable {
		Color3f difference = (frame[pixel] - reference[pixel]).head<3>();
		error_buffer[pixel] = difference.abs() / (1e-3f + reference[pixel].head<3>());
	});
	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return 100.f * val.mean(); }, 0.f,
			   thrust::plus<float>()) / n_elements;
}

float calc_metric_relmse(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	Color3f *error_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable {
		Color difference = (frame[pixel] - reference[pixel]).head<3>().abs();
		for (int ch = 0; ch < Color::dim; ch++) {
			error_buffer[pixel][ch] =
				reference[pixel][ch] == 0.f ? 0.f : difference[ch] / reference[pixel][ch];	
		}
	});
	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { 
					return val.pow(2).mean(); 
				}, 0.f,
			   thrust::plus<float>()) / n_elements;
}

KRR_CALLABLE Color mse(const Color& y, const Color& ref) {
	return (y - ref).abs().pow(2); 
}

KRR_CALLABLE Color mape(const Color &y, const Color &ref) { 
	return (y - ref).abs() / (ref + 0.01); 
}

KRR_CALLABLE Color smape(const Color &y, const Color &ref) { 
	return (y - ref).abs() / (ref + y + 0.01); 
}

// is in fact MRSE...
KRR_CALLABLE Color rel_mse(const Color &y, const Color &ref) {
	Color ret{}, diff = (y - ref).abs();
	for (int ch = 0; ch < Color::dim; ch++) {
		ret[ch] = ref[ch] == 0.f ? 0.f : pow2(diff[ch] / ref[ch]);
	}
	return ret;
}

float calc_metric(const Color4f *frame, const Color4f *reference, 
	size_t n_elements, ErrorMetric metric) {
	Color3f *error_buffer = initialize_metric(n_elements);
	thrust::transform(thrust::device, frame, frame + n_elements, reference, error_buffer,
		[=] KRR_DEVICE(Color y, Color ref) -> Color {
#if METRIC_IN_SRGB
			y = utils::linear2srgb(y);
			ref = utils::linear2srgb(ref);
#endif
			switch (metric) {
				case ErrorMetric::MSE:
					return mse(y, ref);
				case ErrorMetric::MAPE:
					return mape(y, ref);
				case ErrorMetric::SMAPE:
					return smape(y, ref);
				case ErrorMetric::RelMSE:
				default:
					return rel_mse(y, ref);
			}
	});
	
	return thrust::transform_reduce(
			   thrust::device, error_buffer, error_buffer + n_elements,
				[] KRR_DEVICE(const Color3f &val) -> float {
					return val.mean(); }, 0.f, thrust::plus<float>()) / n_elements;
}

KRR_NAMESPACE_END