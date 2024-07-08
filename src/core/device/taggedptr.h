// Code taken and modified from pbrt-v4,
// Originally licensed under the Apache License, Version 2.0.
#pragma once
#include "types.h"

NAMESPACE_BEGIN(krr)

template <typename... Ts> class TaggedPointer {
public:
	using Types = TypePack<Ts...>;

	TaggedPointer() = default;

	template <typename T> KRR_CALLABLE TaggedPointer(T *ptr) {
		uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
		DCHECK_EQ(iptr & ptrMask, iptr);
		constexpr unsigned int type = typeIndex<T>();
		bits						= iptr | ((uintptr_t) type << tagShift);
	}

	KRR_CALLABLE TaggedPointer(void *ptr, unsigned int type) {
		uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
		DCHECK_EQ(iptr & ptrMask, iptr);
		DCHECK_LE(type, maxTag());
		bits = iptr | ((uintptr_t) type << tagShift);
	}

	KRR_CALLABLE TaggedPointer(std::nullptr_t np) {}

	KRR_CALLABLE TaggedPointer(const TaggedPointer &t) { bits = t.bits; }

	KRR_CALLABLE TaggedPointer &operator=(const TaggedPointer &t) {
		bits = t.bits;
		return *this;
	}

	template <typename T> KRR_CALLABLE static constexpr unsigned int typeIndex() {
		using Tp = typename std::remove_cv_t<T>;
		if constexpr (std::is_same_v<Tp, std::nullptr_t>) return 0;
		return 1 + IndexOf<Tp, Types>::count;
	}

	// the index of the type pack plus one. tag=0 means nullptr.
	KRR_CALLABLE unsigned int tag() const { return ((bits & tagMask) >> tagShift); }
	template <typename T> KRR_CALLABLE bool is() const { return tag() == typeIndex<T>(); }

	KRR_CALLABLE static constexpr unsigned int maxTag() { return sizeof...(Ts); }
	KRR_CALLABLE static constexpr unsigned int numTags() { return maxTag() + 1; }

	KRR_CALLABLE explicit operator bool() const { return (bits & ptrMask) != 0; }

	KRR_CALLABLE bool operator<(const TaggedPointer &tp) const { return bits < tp.bits; }

	template <typename T> KRR_CALLABLE T *cast() {
		DCHECK(is<T>());
		return reinterpret_cast<T *>(ptr());
	}

	template <typename T> KRR_CALLABLE const T *cast() const {
		DCHECK(is<T>());
		return reinterpret_cast<const T *>(ptr());
	}

	template <typename T> KRR_CALLABLE T *castOrNullptr() {
		if (is<T>())
			return reinterpret_cast<T *>(ptr());
		else
			return nullptr;
	}

	template <typename T> KRR_CALLABLE const T *castOrNullptr() const {
		if (is<T>())
			return reinterpret_cast<const T *>(ptr());
		else
			return nullptr;
	}

	KRR_CALLABLE bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }

	KRR_CALLABLE bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

	KRR_CALLABLE void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }

	KRR_CALLABLE const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }

	template <typename F> KRR_CALLABLE decltype(auto) dispatch(F &&func) {
		DCHECK(ptr());
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, ptr(), tag() - 1);
	}

	KRR_CALLABLE size_t realSize() const {
		auto f = [&](auto *ptr) -> size_t {  return sizeof *ptr; };
		return dispatch(f);
	}

	template <typename F> KRR_CALLABLE decltype(auto) dispatch(F &&func) const {
		DCHECK(ptr());
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, ptr(), tag() - 1);
	}

	template <typename F> KRR_CALLABLE static decltype(auto) dispatch(F &&func, int index) {
		using R = typename ReturnType<F, Ts...>::type;
		return Dispatch<F, R, Ts...>(func, (const void *) nullptr, index);
	}

private:
	static_assert(sizeof(uintptr_t) == 8, "Expected uintptr_t to be 64 bits");
	static constexpr int tagShift	  = 57;
	static constexpr int tagBits	  = 64 - tagShift;
	static constexpr uint64_t tagMask = ((1ull << tagBits) - 1) << tagShift;
	static constexpr uint64_t ptrMask = ~tagMask;
	uintptr_t bits					  = 0;
};

NAMESPACE_END(krr)