#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <stdint.h>

class SineWaveSimulation {
	float *m_heightMap;
	size_t m_width, m_height;
	int m_blocks, m_threads;

public:
	SineWaveSimulation(size_t width, size_t height);
	~SineWaveSimulation();
	void initSimulation(float *heightMap);
	void stepSimulation(float time, cudaStream_t stream = 0);
	void initCudaLaunchConfig(int device);
	int initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE);

	size_t getWidth() const { return m_width; }
	size_t getHeight() const { return m_height; }
};
