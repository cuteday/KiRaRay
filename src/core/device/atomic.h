#pragma once
#include "common.h"
#include <atomic>

/* This is a simple wrapper with the expection of both CUDA and CPU usage. */
/* This interface is exclusively designed for KiRaRay, and is completely 
	based on my imagination since I am not familiar with CPP's atomic API :-) */

//#define KRR_LEGACY_CUDA_ATOMICS
#ifdef KRR_DEVICE_CODE
#if (__CUDA_ARCH__ < 700)
#define KRR_LEGACY_CUDA_ATOMICS
#else
#include <cuda/atomic>
#endif
#endif

KRR_NAMESPACE_BEGIN

template <typename T> 
class atomic { 
public:
	atomic() = default;

	atomic(const atomic &other)			   = delete;
	atomic &operator=(const atomic &other) = delete;

	KRR_CALLABLE atomic(T val) { 
		store(val);
	}
	
	KRR_CALLABLE T load() const {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
		return m_val;
#else
		return m_val.load(cuda::std::memory_order_relaxed);
#endif
#else
		return m_val.load(std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE void store(const T& val) {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
		m_val = val;
#else
		m_val.store(val, cuda::std::memory_order_relaxed);
#endif
#else
		m_val.store(val, std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE operator T() const {
		return load();
	}
	
	KRR_CALLABLE T exchange(const T &val) {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
		return atomicExch(&m_val, val);
#else
		return m_val.exchange(val, cuda::std::memory_order_relaxed);
#endif
#else
		return m_val.exchange(val, std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE T fetch_add(const T& val) {
#ifdef KRR_DEVICE_CODE
		return atomicAdd((T*) & m_val, val);
#else
		auto cur = m_val.load();
		/* keep trying adding that atomic float, until no one has changed it in between. */
		while (!m_val.compare_exchange_weak(cur, cur + val));
		return cur;
#endif		
	}
	
private:
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
	T m_val{ 0 };
#else
	cuda::atomic<T, cuda::thread_scope_device> m_val{ 0 };
#endif
#else
	std::atomic<T> m_val{ 0 };
#endif
};

KRR_NAMESPACE_END