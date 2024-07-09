#pragma once
#include "common.h"
#include <atomic>
#include <cuda/std/atomic>

/* This is a simple wrapper with the expect of both CUDA and CPU usage. */

NAMESPACE_BEGIN(krr)

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
		return m_val.load(cuda::std::memory_order_relaxed);
#else
		return m_val.load(std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE void store(const T& val) {
#ifdef KRR_DEVICE_CODE
		m_val.store(val, cuda::std::memory_order_relaxed);
#else
		m_val.store(val, std::memory_order_relaxed);
#endif
	}

	KRR_CALLABLE operator T() const {
		return load();
	}
	
	KRR_CALLABLE T exchange(const T &val) {
#ifdef KRR_DEVICE_CODE
		return m_val.exchange(val, cuda::std::memory_order_relaxed);
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
	cuda::std::atomic<T> m_val{0};
#else
	std::atomic<T> m_val{ 0 };
#endif
};

NAMESPACE_END(krr)