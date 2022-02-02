#pragma once
#include <vector>
#include <assert.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "optix7.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
class CUDABuffer {
public:
	CUDABuffer() = default;
	~CUDABuffer() { free(); }

	__both__ inline CUdeviceptr data() const
	{
		return (CUdeviceptr)d_ptr;
	}

	template <typename T>
	__both__  inline T* data() const
	{
		return (T *)d_ptr;
	}

	__both__  inline size_t size() const
	{
		return sizeInBytes;
	}

	//! re-size buffer to given number of bytes
	void resize(size_t size)
	{
		if (sizeInBytes == size) return;
		if (d_ptr) free();
		alloc(size);
	}

	//! allocate to given number of bytes
	void alloc(size_t size)
	{
		assert(d_ptr == nullptr);
		this->sizeInBytes = size;
		CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
	}

	//! free allocated memory
	void free()
	{
		CUDA_CHECK(cudaFree(d_ptr));
		d_ptr = nullptr;
		sizeInBytes = 0;
	}

	template<typename T>
	void alloc_and_copy_from_host(const std::vector<T>& vt)
	{
		alloc(vt.size() * sizeof(T));
		copy_from_host((const T*)vt.data(), vt.size());
	}

	template<typename T>
	void alloc_and_copy_from_device(const std::vector<T>& vt)
	{
		alloc(vt.size() * sizeof(T));
		copy_from_device((const T*)vt.data(), vt.size());
	}

	template<typename T>
	void alloc_and_copy_from_host(const T* t, size_t count)
	{
		alloc(count * sizeof(T));
		copy_from_host((const T*)t, count);
	}

	template<typename T>
	void alloc_and_copy_from_device(const T* t, size_t count)
	{
		alloc(count * sizeof(T));
		copy_from_device((const T*)t, count);
	}

	template<typename T>
	void copy_from_host(const T* t, size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t,
			count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void copy_to_host(T* t, size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy((void*)t, d_ptr,
			count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void copy_from_device(const T* t, size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t,
			count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void copy_to_device(T* t, size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy((void*)t, d_ptr,
			count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

private:
	size_t sizeInBytes{ 0 };
	void* d_ptr{ nullptr };
};

KRR_NAMESPACE_END
