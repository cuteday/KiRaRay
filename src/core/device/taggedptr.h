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

KRR_NAMESPACE_BEGIN

namespace types {

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
			typename T4, typename T5, typename T6, typename T7, typename... Ts>
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

	template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
			typename T4, typename T5, typename T6, typename T7, typename... Ts>
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

}  // namespace detail

// TaggedPointer Definition
template <typename... Ts>
class TaggedPointer {
  public:
	// TaggedPointer Public Types
	using Types = types::TypePack<Ts...>;

	// TaggedPointer Public Methods
	TaggedPointer() = default;

	template <typename T>
	KRR_CALLABLE TaggedPointer(T *ptr) {
		uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
		DCHECK_EQ(iptr & ptrMask, iptr);
		constexpr unsigned int type = typeIndex<T>();
		bits = iptr | ((uintptr_t)type << tagShift);
	}

	KRR_CALLABLE TaggedPointer(std::nullptr_t np) {}

	KRR_CALLABLE TaggedPointer(const TaggedPointer &t) { bits = t.bits; }
	
	KRR_CALLABLE TaggedPointer &operator=(const TaggedPointer &t) {
		bits = t.bits;
		return *this;
	}

	template <typename T>
	KRR_CALLABLE static constexpr unsigned int typeIndex() {
		using Tp = typename std::remove_cv_t<T>;
		if constexpr (std::is_same_v<Tp, std::nullptr_t>)
			return 0;
		return 1 + types::IndexOf<Tp, Types>::count;
	}

	// the index of the type pack plus one. tag=0 means nullptr.
	KRR_CALLABLE unsigned int tag() const { return ((bits & tagMask) >> tagShift); }
	template <typename T>
	KRR_CALLABLE bool is() const {
		return tag() == typeIndex<T>();
	}

	KRR_CALLABLE static constexpr unsigned int maxTag() { return sizeof...(Ts); }
	KRR_CALLABLE static constexpr unsigned int numTags() { return maxTag() + 1; }

	KRR_CALLABLE explicit operator bool() const { return (bits & ptrMask) != 0; }

	KRR_CALLABLE bool operator < (const TaggedPointer &tp) const { return bits < tp.bits; }

	template <typename T>
	KRR_CALLABLE T *cast() {
		DCHECK(is<T>());
		return reinterpret_cast<T *>(ptr());
	}

	template <typename T>
	KRR_CALLABLE const T * cast() const {
		DCHECK(is<T>());
		return reinterpret_cast<const T *>(ptr());
	}

	template <typename T>
	KRR_CALLABLE T *castOrNullptr() {
		if (is<T>())
			return reinterpret_cast<T *>(ptr());
		else
			return nullptr;
	}

	template <typename T>
	KRR_CALLABLE const T *castOrNullptr() const {
		if (is<T>())
			return reinterpret_cast<const T *>(ptr());
		else
			return nullptr;
	}

	KRR_CALLABLE bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
	
	KRR_CALLABLE bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

	KRR_CALLABLE void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }

	KRR_CALLABLE const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }

	template <typename F>
	KRR_CALLABLE decltype(auto) dispatch(F &&func) {
		DCHECK(ptr());
		using R = typename types::ReturnType<F, Ts...>::type;
		return types::Dispatch<F, R, Ts...>(func, ptr(), tag() - 1);
	}

	KRR_CALLABLE size_t realSize() const {
		auto f = [&](auto* ptr)-> size_t {return sizeof * ptr; };
		return dispatch(f);
	}

	template <typename F>
	KRR_CALLABLE decltype(auto) dispatch(F &&func) const {
		DCHECK(ptr());
		using R = typename types::ReturnType<F, Ts...>::type;
		return types::Dispatch<F, R, Ts...>(func, ptr(), tag() - 1);
	}

	template <typename F>
	KRR_CALLABLE static decltype(auto) dispatch(F&& func, int index) {
		using R = typename types::ReturnType<F, Ts...>::type;
		return types::Dispatch<F, R, Ts...>(func, (const void*)nullptr, index);
	}

  private:
	static_assert(sizeof(uintptr_t) == 8, "Expected uintptr_t to be 64 bits");
	static constexpr int tagShift = 57;
	static constexpr int tagBits = 64 - tagShift;
	static constexpr uint64_t tagMask = ((1ull << tagBits) - 1) << tagShift;
	static constexpr uint64_t ptrMask = ~tagMask;
	uintptr_t bits = 0;
};

KRR_NAMESPACE_END