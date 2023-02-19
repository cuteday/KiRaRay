#include "SineWaveSimulation.h"
#include <algorithm>
#include <common.h>
#include <util/check.h>
#include <device/cuda.h>

KRR_NAMESPACE_BEGIN

__global__ void draw_screen(cudaSurfaceObject_t frame, float time, 
						unsigned int width, unsigned int height) {
	unsigned int tid	   = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= width * height) return;
	unsigned int y = tid / width;
	unsigned int x = tid - y * width;

	float4 color = {(int(x + time * 100) % 256) / 256.f, 
					(int(y + time * 100) % 256) / 256.f,
					(int(x + y + time * 100) % 256) / 256.f, 1.f};
	//float4 color = {(float(x) / width), (float(y) / height), 1, 1};
	//float4 orig_color;
	//surf2Dread(&orig_color, frame, x * sizeof(float4), y);
	//if (orig_color.w == 0.f) 
		surf2Dwrite(color, frame, x * sizeof(float4), y);
}

void drawScreen(cudaSurfaceObject_t frame, float time, unsigned int width,
				unsigned int height){
	constexpr int n_threads = 128;
	
	draw_screen<<<width * height / n_threads, n_threads, 0, 0>>>(frame, time, width, height);
}


__global__ void sinewave(float *heightMap, unsigned int width, unsigned int height, float time) {
	const float freq	= 4.0f;
	const size_t stride = gridDim.x * blockDim.x;

	// Iterate through the entire array in a way that is
	// independent of the grid configuration
	for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < width * height; tid += stride) {
		// Calculate the x, y coordinates
		const size_t y = tid / width;
		const size_t x = tid - y * width;
		// Normalize x, y to [0,1]
		const float u = ((2.0f * x) / width) - 1.0f;
		const float v = ((2.0f * y) / height) - 1.0f;
		// Calculate the new height value
		const float w = 0.5f * sinf(u * freq + time) * cosf(v * freq + time);
		// Store this new height value
		heightMap[tid] = w;
	}
}

SineWaveSimulation::SineWaveSimulation(size_t width, size_t height)
	: m_heightMap(nullptr), m_width(width), m_height(height) {
}

void SineWaveSimulation::initCudaLaunchConfig(int device) {
	cudaDeviceProp prop = {};
	CUDA_CHECK(cudaSetDevice(device));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

	// We don't need large block sizes, since there's not much inter-thread
	// communication
	m_threads = prop.warpSize;

	// Use the occupancy calculator and fill the gpu as best as we can
	CUDA_CHECK(
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_blocks, sinewave, prop.warpSize, 0));
	m_blocks *= prop.multiProcessorCount;

	// Go ahead and the clamp the blocks to the minimum needed for this
	// height/width
	m_blocks = std::min(m_blocks, (int) ((m_width * m_height + m_threads - 1) / m_threads));
}

SineWaveSimulation::~SineWaveSimulation() {
	m_heightMap = NULL;
}

void SineWaveSimulation::initSimulation(float *heights) {
	m_heightMap = heights;
}

void SineWaveSimulation::stepSimulation(float time, cudaStream_t stream) {
	sinewave<<<m_blocks, m_threads, 0, stream>>>(m_heightMap, m_width, m_height, time);
	CUDA_SYNC_CHECK();
}

KRR_NAMESPACE_END