#pragma once
#include <vector>
#include <assert.h>

#include "device/optix.h"
#include "common.h"
#include "util/check.h"

KRR_NAMESPACE_BEGIN

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
class CUDABuffer {
public:
	CUDABuffer() = default;
	~CUDABuffer() { free(); }

	KRR_CALLABLE CUdeviceptr data() const
	{
		return (CUdeviceptr)d_ptr;
	}

	KRR_CALLABLE size_t size() const
	{
		return sizeInBytes;
	}

	//! re-size buffer to given number of bytes
	void resize(size_t size)
	{
		if (sizeInBytes == size) return;
		if (d_ptr) free();
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
		if (vt.size() == 0) return;
		resize(vt.size() * sizeof(T));
		copy_from_host((const T*)vt.data(), vt.size());
	}

	template<typename T>
	void alloc_and_copy_from_device(const std::vector<T>& vt)
	{
		if (vt.size() == 0) return;
		resize(vt.size() * sizeof(T));
		copy_from_device((const T*)vt.data(), vt.size());
	}

	template<typename T>
	void alloc_and_copy_from_host(const T* t, size_t count)
	{
		if (count == 0) return;
		resize(count * sizeof(T));
		copy_from_host((const T*)t, count);
	}

	template<typename T>
	void alloc_and_copy_from_device(const T* t, size_t count)
	{
		if (count == 0) return;
		resize(count * sizeof(T));
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

template <typename T>
class TypedBuffer {
public:
	KRR_CALLABLE T* data() const { return d_ptr; }

	KRR_CALLABLE T& operator [] (uint index) { 
		assert(index < m_size);
		return d_ptr[index]; 
	}

	KRR_CALLABLE size_t size() const { return m_size; }

	KRR_CALLABLE size_t sizeInBytes() const { return m_size * sizeof(T); }

	inline void resize(size_t new_size) {
		if (m_size == new_size) return;
		if (d_ptr) clear();
		m_size = new_size;
		CUDA_CHECK(cudaMalloc((void**)&d_ptr, new_size * sizeof(T)));
	}

	inline void clear(){
		CUDA_CHECK(cudaFree(d_ptr));
		d_ptr = nullptr;
		m_size = 0;
	}

	void alloc_and_copy_from_host(const std::vector<T>& vt)
	{
		if (vt.size() == 0) return;
		resize(vt.size());
		copy_from_host((const T*)vt.data(), vt.size());
	}

	void alloc_and_copy_from_device(const std::vector<T>& vt)
	{
		if (vt.size() == 0) return;
		resize(vt.size());
		copy_from_device((const T*)vt.data(), vt.size());
	}

	void alloc_and_copy_from_host(const T* t, size_t count)
	{
		if (count == 0) return;
		resize(count * sizeof(T));
		copy_from_host((const T*)t, count);
	}

	void alloc_and_copy_from_device(const T* t, size_t count)
	{
		if (count == 0) return;
		resize(count * sizeof(T));
		copy_from_device((const T*)t, count);
	}

	void copy_from_host(const T* t, size_t count)
	{
		assert(d_ptr);
		assert(m_size >= count);
		CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t,
			count * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copy_to_host(T* t, size_t count)
	{
		assert(d_ptr);
		assert(m_size >= count);
		CUDA_CHECK(cudaMemcpy((void*)t, d_ptr,
			count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void copy_from_device(const T* t, size_t count)
	{
		assert(d_ptr);
		assert(m_size >= count);
		CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t,
			count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	void copy_to_device(T* t, size_t count)
	{
		assert(d_ptr);
		assert(m_size >= count);
		CUDA_CHECK(cudaMemcpy((void*)t, d_ptr,
			count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

private:
	size_t m_size;
	T* d_ptr{ nullptr };
};

KRR_NAMESPACE_END
