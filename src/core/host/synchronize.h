#include "common.h"

KRR_NAMESPACE_BEGIN

class SpinLock {
public:
	SpinLock() {
		m_mutex.clear(std::memory_order_release);
	}

	SpinLock(const SpinLock& other) { m_mutex.clear(std::memory_order_release); }
	SpinLock& operator=(const SpinLock& other) { return *this; }

	void lock() {
		while (m_mutex.test_and_set(std::memory_order_acquire)) {}
	}

	void unlock() {
		m_mutex.clear(std::memory_order_release);
	}
private:
	std::atomic_flag m_mutex;
};

KRR_NAMESPACE_END