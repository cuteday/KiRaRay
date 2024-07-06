// Code taken and modified from pbrt-v4,
// Originally licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <string>
#include <functional>
#include <type_traits>
#include <iterator>
#include <shared_mutex>
#include <tuple>

#include "common.h"
#include "util/check.h"

NAMESPACE_BEGIN(krr)

template <typename... Ts>
struct TypePack {
	static constexpr size_t count = sizeof...(Ts);
};

template <typename T, typename... Ts>
struct IndexOf {
	static constexpr int count = 0;
	static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> {
	static constexpr int count = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> {
	static constexpr int count = 1 + IndexOf<T, TypePack<Ts...>>::count;
};

template <typename T, typename... Ts>
struct HasType {
	static constexpr bool value = false;
};

template <typename T, typename Tfirst, typename... Ts>
struct HasType<T, TypePack<Tfirst, Ts...>> {
	static constexpr bool value =
		(std::is_same<T, Tfirst>::value || HasType<T, TypePack<Ts...>>::value);
};

template <typename T>
struct GetFirst {};
template <typename T, typename... Ts>
struct GetFirst<TypePack<T, Ts...>> {
	using type = T;
};

template <typename T>
struct RemoveFirst {};
template <typename T, typename... Ts>
struct RemoveFirst<TypePack<T, Ts...>> {
	using type = TypePack<Ts...>;
};

template <int index, typename T, typename... Ts>
struct RemoveFirstN;
template <int index, typename T, typename... Ts>
struct RemoveFirstN<index, TypePack<T, Ts...>> {
	using type = typename RemoveFirstN<index - 1, TypePack<Ts...>>::type;
};

template <typename T, typename... Ts>
struct RemoveFirstN<0, TypePack<T, Ts...>> {
	using type = TypePack<T, Ts...>;
};

template <typename... Ts>
struct Prepend;
template <typename T, typename... Ts>
struct Prepend<T, TypePack<Ts...>> {
	using type = TypePack<T, Ts...>;
};
template <typename... Ts>
struct Prepend<void, TypePack<Ts...>> {
	using type = TypePack<Ts...>;
};

template <int index, typename T, typename... Ts>
struct TakeFirstN;
template <int index, typename T, typename... Ts>
struct TakeFirstN<index, TypePack<T, Ts...>> {
	using type =
		typename Prepend<T, typename TakeFirstN<index - 1, TypePack<Ts...>>::type>::type;
};
template <typename T, typename... Ts>
struct TakeFirstN<1, TypePack<T, Ts...>> {
	using type = TypePack<T>;
};

template <template <typename> class M, typename... Ts>
struct MapType;
template <template <typename> class M, typename T>
struct MapType<M, TypePack<T>> {
	using type = TypePack<M<T>>;
};

template <template <typename> class M, typename T, typename... Ts>
struct MapType<M, TypePack<T, Ts...>> {
	using type = typename Prepend<M<T>, typename MapType<M, TypePack<Ts...>>::type>::type;
};

template <typename T, typename... Ts> 
struct MaximumSizeOfTypePack {
	static constexpr size_t value = sizeof(T);
};
template <typename T, typename... Ts> 
struct MaximumSizeOfTypePack<TypePack<T, Ts...>> {
	static constexpr size_t value = std::max(sizeof(T), MaximumSizeOfTypePack<TypePack<Ts...>>::value);
};

template <typename T>
constexpr size_t MaximumSizeOfTypes() {
	return sizeof(T);
}

template <typename T, typename... Ts>
constexpr size_t MaximumSizeOfTypes() {
	return std::max(sizeof(T), MaximumSizeOfTypes<Ts...>());
}

// TaggedPointer Helper Templates
template <typename F, typename R, typename T>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_EQ(0, index);
	return func((const T *)ptr);
}

template <typename F, typename R, typename T>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_EQ(0, index);
	return func((T *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 2);

	if (index == 0)
		return func((const T0 *)ptr);
	else
		return func((const T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 2);

	if (index == 0)
		return func((T0 *)ptr);
	else
		return func((T1 *)ptr);
}

template <typename F, typename R, typename T0, typename T1, typename T2>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 3);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	default:
		return func((const T2 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 3);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	default:
		return func((T2 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 4);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	case 2:
		return func((const T2 *)ptr);
	default:
		return func((const T3 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 4);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	default:
		return func((T3 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 5);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	case 2:
		return func((const T2 *)ptr);
	case 3:
		return func((const T3 *)ptr);
	default:
		return func((const T4 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 5);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	case 3:
		return func((T3 *)ptr);
	default:
		return func((T4 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 6);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	case 2:
		return func((const T2 *)ptr);
	case 3:
		return func((const T3 *)ptr);
	case 4:
		return func((const T4 *)ptr);
	default:
		return func((const T5 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 6);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	case 3:
		return func((T3 *)ptr);
	case 4:
		return func((T4 *)ptr);
	default:
		return func((T5 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5, typename T6>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 7);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	case 2:
		return func((const T2 *)ptr);
	case 3:
		return func((const T3 *)ptr);
	case 4:
		return func((const T4 *)ptr);
	case 5:
		return func((const T5 *)ptr);
	default:
		return func((const T6 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5, typename T6>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 7);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	case 3:
		return func((T3 *)ptr);
	case 4:
		return func((T4 *)ptr);
	case 5:
		return func((T5 *)ptr);
	default:
		return func((T6 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5, typename T6, typename T7>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 8);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	case 3:
		return func((T3 *)ptr);
	case 4:
		return func((T4 *)ptr);
	case 5:
		return func((T5 *)ptr);
	case 6:
		return func((T6 *)ptr);
	default:
		return func((T7 *)ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
		typename T4, typename T5, typename T6, typename T7>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);
	DCHECK_LT(index, 8);

	switch (index) {
	case 0:
		return func((const T0 *) ptr);
	case 1:
		return func((const T1 *) ptr);
	case 2:
		return func((const T2 *) ptr);
	case 3:
		return func((const T3 *) ptr);
	case 4:
		return func((const T4 *) ptr);
	case 5:
		return func((const T5 *) ptr);
	case 6:
		return func((const T6 *) ptr);
	default:
		return func((const T7 *) ptr);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3, typename T4,
		  typename T5, typename T6, typename T7, typename... Ts,
		  typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
KRR_CALLABLE R Dispatch(F &&func, const void *ptr, int index) {
	DCHECK_GE(index, 0);

	switch (index) {
	case 0:
		return func((const T0 *)ptr);
	case 1:
		return func((const T1 *)ptr);
	case 2:
		return func((const T2 *)ptr);
	case 3:
		return func((const T3 *)ptr);
	case 4:
		return func((const T4 *)ptr);
	case 5:
		return func((const T5 *)ptr);
	case 6:
		return func((const T6 *)ptr);
	case 7:
		return func((const T7 *)ptr);
	default:
		return Dispatch<F, R, Ts...>(func, ptr, index - 8);
	}
}

template <typename F, typename R, typename T0, typename T1, typename T2, typename T3, typename T4,
		  typename T5, typename T6, typename T7, typename... Ts,
		  typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
KRR_CALLABLE R Dispatch(F &&func, void *ptr, int index) {
	DCHECK_GE(index, 0);

	switch (index) {
	case 0:
		return func((T0 *)ptr);
	case 1:
		return func((T1 *)ptr);
	case 2:
		return func((T2 *)ptr);
	case 3:
		return func((T3 *)ptr);
	case 4:
		return func((T4 *)ptr);
	case 5:
		return func((T5 *)ptr);
	case 6:
		return func((T6 *)ptr);
	case 7:
		return func((T7 *)ptr);
	default:
		return Dispatch<F, R, Ts...>(func, ptr, index - 8);
	}
}

template <typename... Ts>
struct IsSameType;
template <>
struct IsSameType<> {
	static constexpr bool value = true;
};
template <typename T>
struct IsSameType<T> {
	static constexpr bool value = true;
};

template <typename T, typename U, typename... Ts>
struct IsSameType<T, U, Ts...> {
	static constexpr bool value =
		(std::is_same<T, U>::value && IsSameType<U, Ts...>::value);
};

template <typename... Ts>
struct SameType;
template <typename T, typename... Ts>
struct SameType<T, Ts...> {
	using type = T;
	static_assert(IsSameType<T, Ts...>::value, "Not all types in pack are the same");
};

template <typename F, typename... Ts>
struct ReturnType {
	using type = typename SameType<typename std::invoke_result_t<F, Ts *>...>::type;
};

template <typename F, typename... Ts>
struct ReturnTypeConst {
	using type = typename SameType<typename std::invoke_result_t<F, const Ts *>...>::type;
};


NAMESPACE_END(krr)