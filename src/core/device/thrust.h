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
	using block_key_type = std::pair<std::ptrdiff_t, std::size_t>;	// size, alignment
	using free_blocks_container = std::multimap<block_key_type, void_ptr>;
	using allocated_blocks_container = std::vector<std::pair<void_ptr, block_key_type>>;

	free_blocks_container free_blocks;
	allocated_blocks_container allocated_blocks;
	Upstream *m_upstream;

public:
	void release() {
		Log(Info, "thrust_cached_resource::release()");
		// Deallocate all outstanding blocks in both lists.
		for (typename free_blocks_container::iterator i = free_blocks.begin(); i != free_blocks.end(); ++i)
			m_upstream->do_deallocate(i->second, i->first.first, i->first.second);

		for (typename allocated_blocks_container::iterator i = allocated_blocks.begin();
			 i != allocated_blocks.end(); ++i)
			m_upstream->do_deallocate(i->first, i->second.first, i->second.second);
	}

	void_ptr do_allocate(std::size_t bytes, std::size_t alignment) override {
		Log(Info, "thrust_cached_resource::do_allocate(): num_bytes == %zu", bytes);
		void_ptr result = nullptr;

		typename free_blocks_container::iterator free_block = free_blocks.find({bytes, alignment});

		if (free_block != free_blocks.end()) {
			Log(Info, "thrust_cached_resource::do_allocate(): found a free block of %zd bytes", bytes);
			result = free_block->second;
			free_blocks.erase(free_block);
		} else {
			Log(Info, "thrust_cached_resource::do_allocate(): allocating new block of %zd bytes", bytes);
			result = m_upstream->do_allocate(bytes, alignment);
		}

		allocated_blocks.push_back(std::make_pair(result, block_key_type{bytes, alignment}));
		return result;
	}

	void do_deallocate(void_ptr ptr, std::size_t bytes, std::size_t alignment) override {
		Log(Info, "thrust_cached_resource::do_deallocate(): ptr == %p", reinterpret_cast<void *>(ptr.get()));

		//typename allocated_blocks_container::iterator iter = allocated_blocks.find(ptr);
		typename allocated_blocks_container::iterator iter = std::find_if(allocated_blocks.begin(), 
			allocated_blocks.end(), [ptr](const typename allocated_blocks_container::value_type& pair){
							 return pair.first == ptr; });
		if (iter == allocated_blocks.end()) {
			Log(Error, "Pointer `%p` was not allocated by this allocator",
				reinterpret_cast<void *>(ptr.get()));
			return;
		}

		block_key_type key = iter->second;

		allocated_blocks.erase(iter);
		free_blocks.insert(std::make_pair(key, ptr));
	}
};

NAMESPACE_END(krr)