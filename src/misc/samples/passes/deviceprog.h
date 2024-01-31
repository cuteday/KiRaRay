#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <common.h>

NAMESPACE_BEGIN(krr)

void drawScreen(CUstream stream, cudaSurfaceObject_t frame, float time, unsigned int width,
							unsigned int height);

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

NAMESPACE_END(krr)