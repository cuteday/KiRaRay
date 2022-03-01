// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#include "interop.h"

KRR_NAMESPACE_BEGIN

namespace inter {

	class NewDeleteResource : public memory_resource {
		void* do_allocate(size_t size, size_t alignment) {
			return _aligned_malloc(size, alignment);
		}

		void do_deallocate(void* ptr, size_t bytes, size_t alignment) {
			if (!ptr)
				return;
#if 1
			_aligned_free(ptr);
#else
			free(ptr);
#endif
		}

		bool do_is_equal(const memory_resource& other) const noexcept {
			return this == &other;
		}
	};

	static NewDeleteResource* ndr;

	memory_resource* new_delete_resource() noexcept {
		if (!ndr)
			ndr = new NewDeleteResource;
		return ndr;
	}

	static memory_resource* defaultMemoryResource = new_delete_resource();

	memory_resource* set_default_resource(memory_resource* r) noexcept {
		memory_resource* orig = defaultMemoryResource;
		defaultMemoryResource = r;
		return orig;
	}

	memory_resource* get_default_resource() noexcept {
		return defaultMemoryResource;
	}

}

KRR_NAMESPACE_END