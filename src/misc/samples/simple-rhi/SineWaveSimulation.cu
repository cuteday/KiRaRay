#include "SineWaveSimulation.h"
#include <algorithm>
#include <common.h>
#include <util/check.h>
#include <device/cuda.h>

KRR_NAMESPACE_BEGIN

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return ((unsigned int) (rgba.w * 255.0f) << 24) | ((unsigned int) (rgba.z * 255.0f) << 16) |
		   ((unsigned int) (rgba.y * 255.0f) << 8) | ((unsigned int) (rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c) {
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;		   //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
	return rgba;
}

__global__ void draw_screen(size_t n_elements, cudaSurfaceObject_t frame, float time, 
						unsigned int width, unsigned int height) {
	size_t tid	   = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;
	const size_t y = tid / width;
	const size_t x = tid - y * width;

	//float4 color = {(int(x + time * 100) % 256) / 256.f, (int(y + time * 100) % 256) / 256.f,
	//				 (int(x + y + time * 100) % 256) / 256.f, 1.f};
	float4 color = {(float(x) / width), (float(y) / height), 1, 1};
	unsigned int data = rgbaFloatToInt(color);
	//uchar4 color = {102, 204, 255, 255};
	surf2Dwrite(data, frame, x * sizeof(uchar4), y);
}

void drawScreen(cudaSurfaceObject_t frame, float time, unsigned int width,
				unsigned int height){
	LinearKernel(draw_screen, 0, width * height, frame, time, width, height);
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