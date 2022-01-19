#pragma once

#include "optix7.h"
// common std stuff
#include <vector>
#include <assert.h>

namespace krr {

  /*! simple wrapper for creating, and managing a device-side CUDA
      buffer */
  struct CUDABuffer {
    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
      if (d_ptr) free();
      alloc(size);
    }
    
    //! allocate to given number of bytes
    void alloc(size_t size)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, sizeInBytes));
    }

    //! free allocated memory
    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
    }

    template<typename T>
    void alloc_and_copy_from_host(const std::vector<T> &vt)
    {
      alloc(vt.size()*sizeof(T));
      copy_from_host((const T*)vt.data(),vt.size());
    }
    
    template<typename T>
    void copy_from_host(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes >= count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void copy_to_host(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes >= count*sizeof(T));
      CUDA_CHECK(Memcpy((void *)t, d_ptr,
                        count*sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void copy_from_device(const T* t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes >= count * sizeof(T));
        CUDA_CHECK(Memcpy(d_ptr, (void*)t,
            count * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    template<typename T>
    void copy_to_device(T* t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes >= count * sizeof(T));
        CUDA_CHECK(Memcpy((void*)t, d_ptr,
            count * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };

} // ::osc
