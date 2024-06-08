#pragma once
#include "common.h"
#include <atomic>
#ifdef __NVCC__
#include <thrust/sort.h>
#endif

#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"

NAMESPACE_BEGIN(krr)
template <typename WorkItem> class WorkQueue : public SOA<WorkItem> {
public:
	using value_type			 = WorkItem;
	using iterator				 = typename SOA<WorkItem>::iterator;
	using const_iterator		 = typename SOA<WorkItem>::const_iterator;
	using reverse_iterator		 = typename SOA<WorkItem>::reverse_iterator;
	using const_reverse_iterator = typename SOA<WorkItem>::const_reverse_iterator;

	WorkQueue() = default;
	KRR_HOST WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
	KRR_HOST WorkQueue &operator=(const WorkQueue &w) {
		SOA<WorkItem>::operator=(w);
		m_size.store(w.m_size);
		return *this;
	}

	KRR_CALLABLE iterator begin() { return iterator(this, 0); }
	KRR_CALLABLE const_iterator begin() const { return const_iterator(this, 0); }
	KRR_CALLABLE iterator end() { return iterator(this, m_size.load()); }
	KRR_CALLABLE const_iterator end() const { return const_iterator(this, m_size.load()); }

	KRR_CALLABLE int size() const { return m_size.load(); }
	KRR_CALLABLE int capacity() const { return nAlloc; }
	KRR_CALLABLE void reset() { m_size.store(0); }

	KRR_CALLABLE int push(const WorkItem &w) {
		int index	   = allocateEntry();
		(*this)[index] = w;
		return index;
	}

	template <typename... Args> KRR_CALLABLE int emplace(Args &&...args) {
		int index = allocateEntry();
		new (&(*this)[index]) WorkItem(std::forward<Args>(args)...);
		return index;
	}

protected:
	KRR_CALLABLE int allocateEntry() { return m_size.fetch_add(1); }

	atomic<int> m_size{0};
};

template <typename WorkItem, typename Key> class SortableWorkQueue : public WorkQueue<WorkItem> {
public:
	using KeyType		= Key;
	SortableWorkQueue() = default;
	KRR_HOST SortableWorkQueue(int n, Allocator alloc) : WorkQueue<WorkItem>(n, alloc) {
		m_keys = TypedBuffer<Key>(n);
	}

	KRR_HOST SortableWorkQueue &operator=(const SortableWorkQueue &w) {
		WorkQueue<WorkItem>::operator=(w);
		m_keys = w.m_keys;
		return *this;
	}

	KRR_CALLABLE TypedBuffer<Key> &keys() { return m_keys; }

	template <typename F>
	void updateKeys(F mapping, size_t max_elements, const Key &oob_val, CUstream stream) {
		auto *queue = this;
		GPUParallelFor(
			max_elements,
			[=] KRR_DEVICE(int index) {
				if (index >= queue->size())
					queue->keys()[index] = oob_val;
				else
					queue->keys()[index] = mapping(queue->operator[](index));
			},
			stream);
	}

	template <typename Compare> void sort(Compare comp, size_t max_elements, CUstream stream) {
#ifdef __NVCC__
		thrust::sort_by_key(thrust::device.on(stream), m_keys.data(), m_keys.data() + max_elements,
							this->begin(), comp);
#endif
	}

	void resize(int n, Allocator alloc) {
		WorkQueue<WorkItem>::resize(n, alloc);
		m_keys.resize(n);
	}

protected:
	TypedBuffer<Key> m_keys;
};

template <typename T> class MultiWorkQueue;

template <typename... Ts> class MultiWorkQueue<TypePack<Ts...>> {
public:
	template <typename T> KRR_CALLABLE WorkQueue<T> *get() {
		return &gpu::get<WorkQueue<T>>(m_queues);
	}

	MultiWorkQueue(int n, Allocator alloc, gpu::span<const bool> haveType) {
		int index = 0;
		((*get<Ts>() = WorkQueue<Ts>(haveType[index++] ? n : 1, alloc)), ...);
	}

	template <typename T> KRR_CALLABLE int size() const { return get<T>()->size(); }

	template <typename T> KRR_CALLABLE int push(const T &value) { return get<T>()->push(value); }

	template <typename T, typename... Args> KRR_CALLABLE int emplace(Args &&...args) {
		return get<T>()->emplace(std::forward<Args>(args)...);
	}

	KRR_CALLABLE void reset() { (get<Ts>()->reset(), ...); }

private:
	gpu::tuple<WorkQueue<Ts>...> m_queues;
};

// Helper functions and basic classes

template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements, F &&func, CUstream stream = 0);

#ifdef __NVCC__
template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements, F &&func, CUstream stream) {
	GPUParallelFor(
		nElements,
		[=] KRR_DEVICE(int index) mutable {
			if (index >= q->size()) return;
			func((*q)[index]);
		},
		stream);
}
#endif

NAMESPACE_END(krr)