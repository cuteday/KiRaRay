#pragma once
#include "common.h"
#include "device/buffer.h"
#include "device/types.h"
#include "device/gpustd.h"

NAMESPACE_BEGIN(krr)

template <typename T, int Dim> 
class Grid {
public:
	Grid() = default;
	
	TypedBufferView<T> voxels;
	Vector<int, Dim> res;
};

namespace gpu {
template <typename T> class multi_vector;

template <typename... Ts> 
class multi_vector<TypePack<Ts...>> {
public:
	template <typename T> 
	KRR_CALLABLE gpu::vector<T> *get() {
		return &gpu::get<gpu::vector<T>>(m_queues);
	}

	multi_vector(int n, Allocator alloc, gpu::span<const bool> haveType) {
		int index = 0;
		((*get<Ts>() = gpu::vector<Ts>(haveType[index++] ? n : 1, alloc)), ...);
	}

	template <typename T> 
	KRR_CALLABLE int size() const { return get<T>()->size(); }

	template <typename T> 
	KRR_CALLABLE int push_back(const T &value) {
		return get<T>()->push_back(value);
	}

	template <typename T, typename... Args> 
	KRR_CALLABLE int emplace_back(Args &&...args) {
		return get<T>()->emplace_back(std::forward<Args>(args)...);
	}

	KRR_CALLABLE void clear() { (get<Ts>()->clear(), ...); }

private:
	gpu::tuple<gpu::vector<Ts>...> m_queues;
};
}

NAMESPACE_END(krr)