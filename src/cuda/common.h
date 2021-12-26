#pragma once
#include <cuda_runtime.h>

__global__ void spgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);