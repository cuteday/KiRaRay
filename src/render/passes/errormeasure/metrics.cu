// https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"

#include "thrust/reduce.h"
#include "thrust/transform_reduce.h"
#include "thrust/execution_policy.h"

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
	Color3f *diff_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable { 
		Color3f difference = (frame[pixel] - reference[pixel]).head<3>();
		diff_buffer[pixel] = difference.abs();
	});
	return thrust::transform_reduce(thrust::device, diff_buffer, diff_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return val.pow(2).mean(); }, 
		0.f, thrust::plus<float>()) / n_elements;
}

float calc_metric_mape(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	Color3f *mape_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable {
		Color3f difference = (frame[pixel] - reference[pixel]).head<3>();
		mape_buffer[pixel] = difference.abs() / (1e-3f + reference[pixel].head<3>());
	});
	return thrust::transform_reduce(
			   thrust::device, mape_buffer, mape_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return 100.f * val.mean(); }, 0.f,
			   thrust::plus<float>()) / n_elements;
}

float calc_metric_relmse(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	Color3f *mape_buffer = initialize_metric(n_elements);
	GPUParallelFor(n_elements, [=] KRR_DEVICE(int pixel) mutable {
		Color3f difference = (frame[pixel] - reference[pixel]).head<3>();
		mape_buffer[pixel] = difference.abs() / (1e-3f + reference[pixel].head<3>());
	});
	return thrust::transform_reduce(
			   thrust::device, mape_buffer, mape_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return val.pow(2).mean(); }, 0.f,
			   thrust::plus<float>()) / n_elements;
}

KRR_NAMESPACE_END