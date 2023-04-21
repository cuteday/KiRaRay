// adapted from https://github.com/Lunarsong/StructureOfArrays
#pragma once

#include <stddef.h>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <vector>

#include <common.h>
#include "atomic.h"
#include "buffer.h"

KRR_NAMESPACE_BEGIN

#define FOR_EACH_ARRAY(expression)                                                   \
	size_t array_index = 0;                                                          \
	int dummy[]		   = {                                                           \
		   (get_array<Elements>(array_index)->expression, ++array_index, 0)...};     \
	(void) dummy

// Structure of Arrays.
template <typename... Elements> 
class SoA {
public:
	// Helper to deduce the type of the Nth element.
	template <size_t N>
	using NthTypeOf =
		typename std::tuple_element<N, std::tuple<Elements...>>::type;
	SoA() = default;
	SoA(size_t size) {
		size_t array_index = 0;
		int dummy_init[]   = {(new (get_array<Elements>(array_index))
								   TypedBuffer<Elements>(),
							   ++array_index, 0)...};
		(void) dummy_init; // avoids unused variable compiler warnings.
	}

	SoA(const SoA &other) : SoA() { *this = other; }

	SoA(SoA &&other) {
		size_t idx		 = 0;
		int dummy_init[] = {
			// placement new https://isocpp.org/wiki/faq/dtors#placement-new
			(new (get_array<Elements>(idx)) TypedBuffer<Elements>(
				 std::move(*other.get_array<Elements>(idx))),
			 ++idx, 0)...};
		(void) dummy_init; // avoids unused variable compiler warnings.

		size_		= other.size_;
		other.size_ = 0;
	}

	virtual ~SoA() {
		FOR_EACH_ARRAY(~TypedBuffer());
	}

	SoA &operator=(const SoA &other) {
		size_t idx		 = 0;
		int dummy_init[] = {
			(*get_array<Elements>(idx) = *other.get_array<Elements>(idx), ++idx,
			 0)...};
		(void) dummy_init; // avoids unused variable compiler warnings.

		size_ = other.size_;
		return *this;
	}

	SoA &operator=(SoA &&other) {
		size_t idx		 = 0;
		int dummy_init[] = {(*get_array<Elements>(idx) =
								 std::move(*other.get_array<Elements>(idx)),
							 ++idx, 0)...};
		(void) dummy_init; // avoids unused variable compiler warnings.

		size_		= other.size_;
		other.size_ = 0;
		return *this;
	}

	// Returns the number of elements in the arrays.
	KRR_CALLABLE size_t size() const { return size_; }

	// Returns |true| if there are no elements in the arrays, |false| otherwise.
	KRR_CALLABLE bool empty() const { return (size_ == 0); }

	KRR_CALLABLE void assign(const size_t index, Elements &&...elements) {
		FOR_EACH_ARRAY(operator[](index) = std::forward<Elements>(elements));
	}

	KRR_CALLABLE void assign(const size_t index, const Elements &...elements) {
		FOR_EACH_ARRAY(operator[](index) = elements);
	}

	// Resizes the arrays to contain |size| elements;
	KRR_HOST void resize(size_t size) {
		FOR_EACH_ARRAY(resize(size));
		size_ = size;
	}

	// Returns a pointer to the |ArrayIndex|th array.
	template<size_t ArrayIndex> 
	KRR_CALLABLE NthTypeOf<ArrayIndex> *array() {
		static_assert(ArrayIndex < kNumArrays,
					  "Requested invalid array index.");

		using ElementType				= NthTypeOf<ArrayIndex>;
		TypedBuffer<ElementType> *array = get_array<ElementType>(ArrayIndex);

		return array->data();
	}

	// Returns a const pointer to the |ArrayIndex|th array.
	template <size_t ArrayIndex> 
	KRR_CALLABLE const NthTypeOf<ArrayIndex> *array() const {
		static_assert(ArrayIndex < kNumArrays,
					  "Requested invalid array index.");

		using ElementType = NthTypeOf<ArrayIndex>;
		const TypedBuffer<ElementType> *array =
			get_array<ElementType>(ArrayIndex);

		return array->data();
	}

	template <size_t ArrayIndex> 
	KRR_CALLABLE NthTypeOf<ArrayIndex> &get(size_t index) {
		static_assert(ArrayIndex < kNumArrays,
					  "Requested invalid array index in get().");

		using ElementType				= NthTypeOf<ArrayIndex>;
		TypedBuffer<ElementType> *array = get_array<ElementType>(ArrayIndex);

		return (*array)[index];
	}

	// Returns a const reference to the |index|th element from the
	// |ArrayIndex|th array as type |ElementType|.
	template <size_t ArrayIndex>
	KRR_CALLABLE const NthTypeOf<ArrayIndex> &get(size_t index) const {
		static_assert(ArrayIndex < kNumArrays,
					  "Requested invalid array index in get().");

		using ElementType = NthTypeOf<ArrayIndex>;
		const TypedBuffer<ElementType> *array =
			get_array<ElementType>(ArrayIndex);

		return (*array)[index];
	}

	// Returns the number of arrays.
	size_t num_arrays() const { return kNumArrays; }

protected:
	template <class Type> 
	KRR_CALLABLE TypedBuffer<Type> *get_array(size_t array_index) {
		return reinterpret_cast<TypedBuffer<Type> *>(&arrays_[array_index]);
	}

	template <class Type>
	KRR_CALLABLE const TypedBuffer<Type> *get_array(size_t array_index) const {
		return reinterpret_cast<const TypedBuffer<Type> *>(
			&arrays_[array_index]);
	}

	using ArrayType = TypedBuffer<void *>;
	static const size_t kNumArrays = sizeof...(Elements);
	size_t size_				   = 0;
	ArrayType arrays_[kNumArrays];
};

template <typename ...Elements>
class SoAQueue : public SoA<Elements...> {
public:
	SoAQueue() = default;
	KRR_HOST SoAQueue(size_t size) : SoA<Elements...>(size) {}
	KRR_HOST SoAQueue& operator=(const SoAQueue& q) {
		SoAQueue<Elements...>::operator=(q);
		m_size.store(q.m_size);
		return *this;
	}

	KRR_CALLABLE int size() const { return m_size.load(); }

	KRR_CALLABLE void reset() { m_size.store(0); }

	KRR_CALLABLE int push(Elements&& ...e) {
		int index = allocate();
		SoA<Elements...>::assign(index, std::forward<Elements>(e)...);
		return index;
	}

protected:
	KRR_CALLABLE int allocate() { return m_size.fetch_add(1); }

private:
	atomic<int> m_size{0};
};

KRR_NAMESPACE_END