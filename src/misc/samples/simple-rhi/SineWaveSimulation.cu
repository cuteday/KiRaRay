#include "SineWaveSimulation.h"
#include <algorithm>
#include <common.h>
#include <util/check.h>

using namespace krr;

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

int SineWaveSimulation::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
	int current_device	   = 0;
	int device_count	   = 0;
	int devices_prohibited = 0;

	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the GPU which is selected by Vulkan
	while (current_device < device_count) {
		cudaGetDeviceProperties(&deviceProp, current_device);

		if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
			// Compare the cuda device UUID with vulkan UUID
			int ret = memcmp((void *) &deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
			if (ret == 0) {
				CUDA_CHECK(cudaSetDevice(current_device));
				CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, current_device));
				printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", current_device,
					   deviceProp.name, deviceProp.major, deviceProp.minor);

				return current_device;
			}

		} else {
			devices_prohibited++;
		}

		current_device++;
	}

	if (devices_prohibited == device_count) {
		fprintf(stderr, "CUDA error:"
						" No Vulkan-CUDA Interop capable GPU found.\n");
		exit(EXIT_FAILURE);
	}

	return -1;
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
