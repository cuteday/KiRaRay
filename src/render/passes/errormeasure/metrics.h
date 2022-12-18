#pragma once
#include "common.h"

KRR_NAMESPACE_BEGIN

float calc_metric_mse(const Color4f *frame, const Color4f *reference, 
						size_t n_elements);

float calc_metric_mape(const Color4f *frame, const Color4f *reference,
						size_t n_elements);

float calc_metric_relmse(const Color4f *frame, const Color4f *reference,
						size_t n_elements);

KRR_NAMESPACE_END