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

// This example demonstrates how to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::sort. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::system::cuda::malloc.
//
// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

NAMESPACE_BEGIN(krr)

// cached_allocator: a simple allocator for caching allocation requests
struct cached_allocator
{
  cached_allocator() {}

  void *allocate(std::ptrdiff_t num_bytes)
  {
    void *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if(free_block != free_blocks.end())
    {
      std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    }
    else
    {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try
      {
        std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

        // allocate memory and convert cuda::pointer to raw pointer
        result = thrust::system::cuda::malloc(num_bytes).get();
      }
      catch(std::runtime_error &e)
      {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(void *ptr)
  {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

  void free_all()
  {
    std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

    // deallocate all outstanding blocks in both lists
    for(free_blocks_type::iterator i = free_blocks.begin();
        i != free_blocks.end();
        ++i)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->second));
    }

    for(allocated_blocks_type::iterator i = allocated_blocks.begin();
        i != allocated_blocks.end();
        ++i)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->first));
    }
  }

  typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
  typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;
};


// the cache is simply a global variable
// XXX ideally this variable is declared thread_local
cached_allocator g_allocator;


// create a tag derived from system::cuda::tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct cached_allocator_tag : thrust::system::cuda::tag {};


// overload get_temporary_buffer on cached_allocator_tag
// its job is to forward allocation requests to g_allocator
template<typename T>
  thrust::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(cached_allocator_tag, std::ptrdiff_t n)
{
  // ask the allocator for sizeof(T) * n bytes
  T* result = reinterpret_cast<T*>(g_allocator.allocate(sizeof(T) * n));

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}


// overload return_temporary_buffer on cached_allocator_tag
// its job is to forward deallocations to g_allocator
// an overloaded return_temporary_buffer should always accompany
// an overloaded get_temporary_buffer
template<typename Pointer>
  void return_temporary_buffer(cached_allocator_tag, Pointer p)
{
  // return the pointer to the allocator
  g_allocator.deallocate(thrust::raw_pointer_cast(p));
}

NAMESPACE_END(krr)