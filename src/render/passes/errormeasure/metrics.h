#pragma once
#include "common.h"

KRR_NAMESPACE_BEGIN

enum class ErrorMetric { MSE, MAPE, SMAPE, RelMSE, Count };

KRR_ENUM_DEFINE(ErrorMetric, {
	{ErrorMetric::MSE, "mse"},
	{ErrorMetric::MAPE, "mape"},
	{ErrorMetric::SMAPE, "smape"},
	{ErrorMetric::RelMSE, "rel_mse"}
});

float calc_metric(const Color4f *frame, const Color4f *reference, size_t n_elements, ErrorMetric metric);

float calc_metric_mse(const Color4f *frame, const Color4f *reference, size_t n_elements);

float calc_metric_mape(const Color4f *frame, const Color4f *reference, size_t n_elements);

float calc_metric_relmse(const Color4f *frame, const Color4f *reference, size_t n_elements);

KRR_NAMESPACE_END