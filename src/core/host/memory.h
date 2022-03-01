// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#pragma once
#include <atomic>

#include "common.h"
#include "interop.h"
#include "util/check.h"

KRR_NAMESPACE_BEGIN

class TrackedMemory : public inter::memory_resource {
  public:
	TrackedMemory(
		inter::memory_resource *source = inter::get_default_resource())
		: source(source) {}

	void *do_allocate(size_t size, size_t alignment) {
		void *ptr = source->allocate(size, alignment);
		uint64_t currentBytes = allocatedBytes.fetch_add(size) + size;
		uint64_t prevMax = maxAllocatedBytes.load(std::memory_order_relaxed);
		while (prevMax < currentBytes &&
			   !maxAllocatedBytes.compare_exchange_weak(prevMax, currentBytes))
			;
		return ptr;
	}
	void do_deallocate(void *p, size_t bytes, size_t alignment) {
		source->deallocate(p, bytes, alignment);
		allocatedBytes -= bytes;
	}

	bool do_is_equal(const memory_resource &other) const noexcept {
		return this == &other;
	}

	size_t CurrentAllocatedBytes() const { return allocatedBytes.load(); }
	size_t MaxAllocatedBytes() const { return maxAllocatedBytes.load(); }

  private:
	inter::memory_resource *source;
	std::atomic<uint64_t> allocatedBytes{0}, maxAllocatedBytes{0};
};

KRR_NAMESPACE_END