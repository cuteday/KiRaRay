#pragma once
#include "common.h"
#include "device/cuda.h"

NAMESPACE_BEGIN(krr)

enum class ErrorMetric { MSE, MAPE, SMAPE, RelMSE, Count };

KRR_ENUM_DEFINE(ErrorMetric, {
	{ErrorMetric::MSE, "mse"},
	{ErrorMetric::MAPE, "mape"},
	{ErrorMetric::SMAPE, "smape"},
	{ErrorMetric::RelMSE, "rel_mse"}
});

float calc_metric(const CudaRenderTarget& frame, const RGBA *reference, size_t n_elements, ErrorMetric metric);

NAMESPACE_END(krr)