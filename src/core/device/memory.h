// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#pragma once

#include "common.h"
#include "interop.h"
#include "optix7.h"
#include "util/check.h"

KRR_NAMESPACE_BEGIN

class CUDAMemoryResource : public inter::memory_resource {
    void *do_allocate(size_t size, size_t alignment){
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CHECK_EQ(0, intptr_t(ptr) % alignment);
    return ptr;
	}
    void do_deallocate(void *p, size_t bytes, size_t alignment) {
    CUDA_CHECK(cudaFree(p));
}

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }
};

class CUDATrackedMemoryResource : public CUDAMemoryResource {
  public:
    void *do_allocate(size_t size, size_t alignment){
    if (size == 0)
        return nullptr;

    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    DCHECK_EQ(0, intptr_t(ptr) % alignment);

    std::lock_guard<std::mutex> lock(mutex);
    allocations[ptr] = size;
    bytesAllocated += size;

    return ptr;
}

    void do_deallocate(void *p, size_t bytes, size_t alignment){
		if (!p)
			return;

		CUDA_CHECK(cudaFree(p));

		std::lock_guard<std::mutex> lock(mutex);
		auto iter = allocations.find(p);
		DCHECK(iter != allocations.end());
		allocations.erase(iter);
		bytesAllocated -= size;
	}
    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

    void PrefetchToGPU() const {
		int deviceIndex;
		CUDA_CHECK(cudaGetDevice(&deviceIndex));

		std::lock_guard<std::mutex> lock(mutex);

		LOG_VERBOSE("Prefetching %d allocations to GPU memory", allocations.size());
		size_t bytes = 0;
		for (auto iter : allocations) {
			CUDA_CHECK(
				cudaMemPrefetchAsync(iter.first, iter.second, deviceIndex, 0 /* stream */));
			bytes += iter.second;
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		LOG_VERBOSE("Done prefetching: %d bytes total", bytes);
	}
	
    size_t BytesAllocated() const { return bytesAllocated; }

    static CUDATrackedMemoryResource singleton;

  private:
    mutable std::mutex mutex;
    std::atomic<size_t> bytesAllocated{};
    std::unordered_map<void *, size_t> allocations;
};

KRR_NAMESPACE_END