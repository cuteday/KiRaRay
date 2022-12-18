#include "metrics.h"
#include "device/cuda.h"
#include "device/buffer.h"

#include "thrust/transform.h"
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
		diff_buffer[pixel] = difference;
	});
	return thrust::transform_reduce(thrust::device, diff_buffer, diff_buffer + n_elements,
			   [] KRR_DEVICE(const Color3f &val) -> float { return val.abs().mean(); }, 
		0.f, thrust::plus<float>()) / n_elements;
}

float calc_metric_mape(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	return 0.0f;
}

float calc_metric_relmse(const Color4f *frame, const Color4f *reference, 
	size_t n_elements) {
	return 0.0f;
}

KRR_NAMESPACE_END