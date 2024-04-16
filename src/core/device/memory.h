// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.

// A useful blog: https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/,
// Prefetching achieves comparable speed to a before-hand cudaMemcpy() op, but only avaliable on Linux.

#pragma once
#include <map>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cuda.h>

#include "common.h"
#include "device/gpustd.h"
#include "util/check.h"

NAMESPACE_BEGIN(krr)

class CUDAMemory : public gpu::memory_resource {
	void *do_allocate(size_t size, size_t alignment) override {
		void *ptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, size));
		CHECK_EQ(0, intptr_t(ptr) % alignment);
		return ptr;
	}

	void do_deallocate(void *p, size_t bytes, size_t alignment) override {
		CUDA_CHECK(cudaFree(p));
	}

	bool do_is_equal(const memory_resource &other) const noexcept override {
		return this == &other;
	}

	void do_release() override {}
};

class CUDATrackedMemory : public CUDAMemory {
public:
	void *do_allocate(size_t size, size_t alignment) override {
		if (size == 0) return nullptr;

		void *ptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, size));
		DCHECK_EQ(0, intptr_t(ptr) % alignment);

		std::lock_guard<std::mutex> lock(mutex);
		allocations[ptr] = size;
		bytesAllocated += size;

		return ptr;
	}

	void do_deallocate(void *p, size_t size, size_t alignment) override {
		if (!p) return;

		CUDA_CHECK(cudaFree(p));

		std::lock_guard<std::mutex> lock(mutex);
		auto iter = allocations.find(p);
		DCHECK_NE(iter, allocations.end());
		allocations.erase(iter);
		bytesAllocated -= size;
	}

	bool do_is_equal(const memory_resource &other) const noexcept override {
		return this == &other;
	}

	void do_release() override {
		std::lock_guard<std::mutex> lock(mutex);
		for (auto iter : allocations) 
			CUDA_CHECK(cudaFree(iter.first));
		allocations.clear();
		bytesAllocated = 0;
	}


	void PrefetchToGPU() const {
		// only linux supports uniform memory prefetching on demand
#if KRR_PLATFORM_LINUX
		static int deviceIndex = 0;
		CUDA_CHECK(cudaGetDevice(&deviceIndex));

		std::lock_guard<std::mutex> lock(mutex);

		Log(Debug, "Prefetching allocations to GPU memory, "	
			"total allocations: %zd" + allocations.size());
		size_t bytes = 0;
		for (auto iter : allocations) {
			CUDA_CHECK(
				cudaMemPrefetchAsync(iter.first, iter.second, 0, 0 /* stream */));
			bytes += iter.second;
		}
#else 
		CUDA_CHECK(cudaDeviceSynchronize());
#endif
	}
	
	size_t BytesAllocated() const { return bytesAllocated; }

	static CUDATrackedMemory singleton;

  private:
	mutable std::mutex mutex;
	std::atomic<size_t> bytesAllocated{};
	std::unordered_map<void *, size_t> allocations;
};

NAMESPACE_END(krr)