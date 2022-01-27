#include <cuda_runtime.h>

__global__ void spgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
	int M, int N, int K){
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(ty < M && tx < N) {
        float c = 0;
        for(int i = 0; i < K; ++i){
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}