#include <thrust/system/cuda/vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>

#include "common.h"
#include "logger.h"

NAMESPACE_BEGIN(krr)

template <class Upstream>
class thrust_cached_resource final :
	public thrust::mr::memory_resource<typename Upstream::pointer> {
public:
	thrust_cached_resource(Upstream *upstream) : m_upstream(upstream) {}
	thrust_cached_resource() : m_upstream(thrust::mr::get_global_resource<Upstream>()) {}
	~thrust_cached_resource() { release(); }

private:
	typedef typename Upstream::pointer void_ptr;
	typedef std::tuple<std::ptrdiff_t, std::size_t, void_ptr> block_t;
	std::vector<block_t> blocks{};
	Upstream *m_upstream;

public:
	void release() {
		Log(Info, "thrust_cached_resource::release()");
		std::for_each(blocks.begin(), blocks.end(), [this](block_t &block) {
			auto &[bytes, alignment, ptr] = block;
			m_upstream->do_deallocate(ptr, bytes, alignment);
		});
	}

	void_ptr do_allocate(std::size_t bytes, std::size_t alignment) override {
		Log(Info, "thrust_cached_resource::do_allocate(): num_bytes == %zu", bytes);

		void_ptr result = nullptr;

		auto const fitting_block =
			std::find_if(blocks.cbegin(), blocks.cend(), [bytes, alignment](block_t const &block) {
				auto &[b_bytes, b_alignment, _] = block;
				return b_bytes == bytes && b_alignment == alignment;
			});

		if (fitting_block != blocks.end()) {
			Log(Info, "thrust_cached_resource::do_allocate(): found a free block of %zd bytes", bytes);
			result = std::get<2>(*fitting_block);
		} else {
			Log(Info, "thrust_cached_resource::do_allocate(): allocating new block of %zd bytes", bytes);
			result = m_upstream->do_allocate(bytes, alignment);
			blocks.emplace_back(bytes, alignment, result);
		}
		return result;
	}

	void do_deallocate(void_ptr ptr, std::size_t bytes, std::size_t alignment) override {
		Log(Info, "thrust_cached_resource::do_deallocate(): ptr == %p",
			reinterpret_cast<void *>(ptr.get()));
		auto const fitting_block =
			std::find_if(blocks.cbegin(), blocks.cend(),
						 [ptr](block_t const &block) { return std::get<2>(block) == ptr; });

		if (fitting_block == blocks.end())
			Log(Error, "Pointer `%p` was not allocated by this allocator",
				thrust::raw_pointer_cast(ptr));
	}
};

NAMESPACE_END(krr)