#pragma once
/* An utility that is based on std::variant and inspired by TaggedPointer.
* It wraps a variant and implements member function dispatching at runtime,
* which is similar to the virtual function mechanism but purposed for device polymorphism.
*/
#include "types.h"

#include <variant>
#include <cuda/std/variant>

NAMESPACE_BEGIN(krr)

namespace gpu {
#ifdef KRR_DEVICE_CODE
	template <class ...Ts>
	using variant = cuda::std::variant<Ts...>;
	using monostate = cuda::std::monostate;
#else
	template <class... Ts>
	using variant = std::variant<Ts...>;
	using monostate = std::monostate;
#endif
}

template <class Variant, size_t I = 0>
KRR_HOST Variant VariantFromIndex(size_t index) {
	/* Constructing a variant with std::in_place_index seems cudac currently does not support? */
	if constexpr (I >= std::variant_size_v<Variant>) 
		return Variant{std::in_place_index<0>};
    return index == 0 ?
		Variant{std::in_place_index<I>} : VariantFromIndex<Variant, I + 1>(index - 1);
}

template <typename V, typename T, typename... Ts>
KRR_CALLABLE void DefaultConstruct(V& var, size_t index) {
	if (index == 0) var = T();
	else if constexpr (sizeof...(Ts) > 0) 
		DefaultConstruct<V, Ts...>(var, index - 1);
	else assert(false);
}

template <typename... Ts> class VariantClass {
public:
	using Types = TypePack<Ts...>;

	VariantClass() = default;

	template <typename T> KRR_CALLABLE VariantClass(T value) {
		static_assert(HasType<T, Types>::value, "Type not present in the type pack!");
		data = value;
	}

	template <typename T> KRR_CALLABLE VariantClass &operator=(T value) {
		static_assert(HasType<T, Types>::value, "Type not present in the type pack!");
		data = value;
		return *this;
	}

	template <typename T> KRR_CALLABLE static constexpr unsigned int typeIndex() {
		using Tp = typename std::remove_cv_t<T>;
		if constexpr (std::is_same_v<Tp, std::nullptr_t>) return 0;
		return 1 + IndexOf<Tp, Types>::count;
	}

	// the index of the type pack plus one. tag=0 means nullptr.
	KRR_CALLABLE int index() const { return static_cast<int>(data.index()); }
	template <typename T> KRR_CALLABLE bool is() const { return index() == typeIndex<T>(); }

	KRR_CALLABLE static constexpr unsigned int maxIndex() { return sizeof...(Ts); }
	KRR_CALLABLE static constexpr unsigned int numIndices() { return maxIndex() + 1; }

	KRR_CALLABLE explicit operator bool() const { return data.index() != 0; }

	template <typename T> KRR_CALLABLE T *get() {
		DCHECK(is<T>());
		return reinterpret_cast<T *>(ptr());
	}

	template <typename T> KRR_CALLABLE const T *get() const {
		DCHECK(is<T>());
		return reinterpret_cast<const T *>(ptr());
	}

	template <typename T> KRR_CALLABLE T *getIf() {
		if (is<T>())
			return reinterpret_cast<T *>(ptr());
		else
			return nullptr;
	}

	template <typename T> KRR_CALLABLE const T *getIf() const {
		if (is<T>())
			return reinterpret_cast<const T *>(ptr());
		else
			return nullptr;
	}

	KRR_CALLABLE void *ptr() { return reinterpret_cast<void *>(&data); }
	KRR_CALLABLE const void *ptr() const { return reinterpret_cast<const void *>(&data); }

	template <typename F> KRR_CALLABLE decltype(auto) dispatch(F &&func) {
		DCHECK(ptr());
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, ptr(), index() - 1);
	}

	template <typename F> KRR_CALLABLE decltype(auto) dispatch(F &&func) const {
		DCHECK(ptr());
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, ptr(), index() - 1);
	}

	template <typename F> KRR_CALLABLE static decltype(auto) dispatch(F &&func, int index) {
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, (const void *) nullptr, index);
	}

protected:
	KRR_CALLABLE void defaultConstruct(size_t index) {
		DefaultConstruct<decltype(data), Ts...>(data, index);
	}

	gpu::variant<gpu::monostate, Ts...> data;
};

NAMESPACE_END(krr)