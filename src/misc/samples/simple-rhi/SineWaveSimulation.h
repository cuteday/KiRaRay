#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <common.h>

KRR_NAMESPACE_BEGIN

template <typename T>
__global__ void draw_screen(Color4f *pixels, float time, unsigned int width, unsigned int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	pixels[y * width + x] = {(int(x + time * 100) % 256) / 256.f,
							 (int(y + time * 100) % 256) / 256.f,
							 (int(x + y + time * 100) % 256) / 256.f, 1.f};
}

class SineWaveSimulation {
	float *m_heightMap;
	size_t m_width, m_height;
	int m_blocks, m_threads;

public:
	SineWaveSimulation() = default;
	SineWaveSimulation(size_t width, size_t height);
	~SineWaveSimulation();
	void initSimulation(float *heightMap);
	void stepSimulation(float time, cudaStream_t stream = 0);
	void initCudaLaunchConfig(int device);

	size_t getWidth() const { return m_width; }
	size_t getHeight() const { return m_height; }
};

KRR_NAMESPACE_END