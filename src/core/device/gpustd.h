// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#pragma once

#include <limits>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "util/check.h"
#include "cuda.h"
#include "logger.h"
#include "common.h"

NAMESPACE_BEGIN(krr)

namespace gpu {

	class memory_resource {
		static constexpr size_t max_align = alignof(std::max_align_t);

	public:
		virtual ~memory_resource() {};
		void* allocate(size_t bytes, size_t alignment = max_align) {
			if (bytes == 0)
				return nullptr;
			return do_allocate(bytes, alignment);
		}
		void deallocate(void* p, size_t bytes, size_t alignment = max_align) {
			if (!p)
				return;
			return do_deallocate(p, bytes, alignment);
		}
		bool is_equal(const memory_resource& other) const noexcept {
			return do_is_equal(other);
		}

	private:
		virtual void* do_allocate(size_t bytes, size_t alignment) = 0;
		virtual void do_deallocate(void* p, size_t bytes, size_t alignment) = 0;
		virtual bool do_is_equal(const memory_resource& other) const noexcept = 0;
	};

	inline bool operator==(const memory_resource& a, const memory_resource& b) noexcept {
		return a.is_equal(b);
	}

	inline bool operator!=(const memory_resource& a, const memory_resource& b) noexcept {
		return !(a == b);
	}

	// global memory resources
	memory_resource* new_delete_resource() noexcept;
	// TODO: memory_resource* null_memory_resource() noexcept;
	memory_resource* set_default_resource(memory_resource* r) noexcept;
	memory_resource* get_default_resource() noexcept;

	template <class Tp = std::byte>
	class polymorphic_allocator {
	public:
		using value_type = Tp;

		polymorphic_allocator() noexcept { memoryResource = get_default_resource(); }
		polymorphic_allocator(memory_resource* r) : memoryResource(r) {}
		polymorphic_allocator(const polymorphic_allocator& other) = default;
		template <class U>
		polymorphic_allocator(const polymorphic_allocator<U>& other) noexcept
			: memoryResource(other.resource()) {}

		polymorphic_allocator& operator=(const polymorphic_allocator& rhs) = delete;

		// member functions
		[[nodiscard]] Tp* allocate(size_t n) {
			return static_cast<Tp*>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
		}
		void deallocate(Tp* p, size_t n) { resource()->deallocate(p, n); }

		void* allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
			return resource()->allocate(nbytes, alignment);
		}
		void deallocate_bytes(void* p, size_t nbytes,
			size_t alignment = alignof(std::max_align_t)) {
			return resource()->deallocate(p, nbytes, alignment);
		}
		template <class T>
		T* allocate_object(size_t n = 1) {
			return static_cast<T*>(allocate_bytes(n * sizeof(T), alignof(T)));
		}
		template <class T>
		void deallocate_object(T* p, size_t n = 1) {
			deallocate_bytes(p, n * sizeof(T), alignof(T));
		}
		template <class T, class... Args>
		T* new_object(Args &&...args) {
			// NOTE: this doesn't handle constructors that throw exceptions...
			T* p = allocate_object<T>();
			construct(p, std::forward<Args>(args)...);
			return p;
		}
		template <class T>
		void delete_object(T* p) {
			destroy(p);
			deallocate_object(p);
		}

		template <class T, class... Args>
		void construct(T* p, Args &&...args) {
			::new ((void*)p) T(std::forward<Args>(args)...);
		}

		template <class T>
		void destroy(T* p) {
			p->~T();
		}

		memory_resource* resource() const { return memoryResource; }

	private:
		memory_resource* memoryResource;
	};

	template <class T1, class T2>
	bool operator==(const polymorphic_allocator<T1>& a,
		const polymorphic_allocator<T2>& b) noexcept {
		return a.resource() == b.resource();
	}

	template <class T1, class T2>
	bool operator!=(const polymorphic_allocator<T1>& a,
		const polymorphic_allocator<T2>& b) noexcept {
		return !(a == b);
	}

	namespace span_internal {

	// Wrappers for access to container data pointers.
	template <typename C>
	KRR_CALLABLE constexpr auto GetDataImpl(C &c, char) noexcept -> decltype(c.data()) {
		return c.data();
	}

	template <typename C>
	KRR_CALLABLE constexpr auto GetData(C &c) noexcept -> decltype(GetDataImpl(c, 0)) {
		return GetDataImpl(c, 0);
	}

	// Detection idioms for size() and data().
	template <typename C>
	using HasSize = std::is_integral<typename std::decay_t<decltype(std::declval<C &>().size())>>;

	// We want to enable conversion from vector<T*> to span<const T* const> but
	// disable conversion from vector<Derived> to span<Base>. Here we use
	// the fact that U** is convertible to Q* const* if and only if Q is the same
	// type or a more cv-qualified version of U.  We also decay the result type of
	// data() to avoid problems with classes which have a member function data()
	// which returns a reference.
	template <typename T, typename C>
	using HasData =
		std::is_convertible<typename std::decay_t<decltype(GetData(std::declval<C &>()))> *,
							T *const *>;

	} // namespace span_internal

	inline constexpr std::size_t dynamic_extent = -1;

	template <typename T> class span {
	public:
		// Used to determine whether a Span can be constructed from a container of
		// type C.
		template <typename C>
		using EnableIfConvertibleFrom =
			typename std::enable_if_t<span_internal::HasData<T, C>::value &&
									  span_internal::HasSize<C>::value>;

		// Used to SFINAE-enable a function when the slice elements are const.
		template <typename U>
		using EnableIfConstView = typename std::enable_if_t<std::is_const_v<T>, U>;

		// Used to SFINAE-enable a function when the slice elements are mutable.
		template <typename U>
		using EnableIfMutableView = typename std::enable_if_t<!std::is_const_v<T>, U>;

		using value_type	 = typename std::remove_cv_t<T>;
		using iterator		 = T *;
		using const_iterator = const T *;

		KRR_CALLABLE
		span() : ptr(nullptr), n(0) {}
		KRR_CALLABLE
		span(T *ptr, size_t n) : ptr(ptr), n(n) {}
		template <size_t N> KRR_CALLABLE span(T (&a)[N]) : span(a, N) {}
		KRR_CALLABLE
		span(std::initializer_list<value_type> v) : span(v.begin(), v.size()) {}

		// Explicit reference constructor for a mutable `span<T>` type. Can be
		// replaced with MakeSpan() to infer the type parameter.
		template <typename V, typename X = EnableIfConvertibleFrom<V>,
				  typename Y = EnableIfMutableView<V>>
		KRR_CALLABLE explicit span(V &v) noexcept : span(v.data(), v.size()) {}

		// Hack: explicit constructors for std::vector to work around warnings
		// about calling a host function (e.g. vector::size()) form a
		// host+device function (the regular span constructor.)
		template <typename V> span(std::vector<V> &v) noexcept : span(v.data(), v.size()) {}
		template <typename V> span(const std::vector<V> &v) noexcept : span(v.data(), v.size()) {}

		// Implicit reference constructor for a read-only `span<const T>` type
		template <typename V, typename X = EnableIfConvertibleFrom<V>,
				  typename Y = EnableIfConstView<V>>
		KRR_CALLABLE constexpr span(const V &v) noexcept : span(v.data(), v.size()) {}

		KRR_CALLABLE
		iterator begin() { return ptr; }
		KRR_CALLABLE
		iterator end() { return ptr + n; }
		KRR_CALLABLE
		const_iterator begin() const { return ptr; }
		KRR_CALLABLE
		const_iterator end() const { return ptr + n; }

		KRR_CALLABLE
		T &operator[](size_t i) {
			DCHECK_LT(i, size());
			return ptr[i];
		}
		KRR_CALLABLE
		const T &operator[](size_t i) const {
			DCHECK_LT(i, size());
			return ptr[i];
		}

		KRR_CALLABLE
		size_t size() const { return n; };
		KRR_CALLABLE
		bool empty() const { return size() == 0; }
		KRR_CALLABLE
		T *data() { return ptr; }
		KRR_CALLABLE
		const T *data() const { return ptr; }

		KRR_CALLABLE
		T front() const { return ptr[0]; }
		KRR_CALLABLE
		T back() const { return ptr[n - 1]; }

		KRR_CALLABLE
		void remove_prefix(size_t count) {
			// assert(size() >= count);
			ptr += count;
			n -= count;
		}
		KRR_CALLABLE
		void remove_suffix(size_t count) {
			// assert(size() > = count);
			n -= count;
		}

		KRR_CALLABLE
		span subspan(size_t pos, size_t count = dynamic_extent) {
			size_t np = count < (size() - pos) ? count : (size() - pos);
			return span(ptr + pos, np);
		}

	private:
		T *ptr;
		size_t n;
	};

	template <typename T, class Allocator = polymorphic_allocator<T>>
	class vector {
	public:
		using value_type = T;
		using allocator_type = Allocator;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using reference = value_type&;
		using const_reference = const value_type&;
		using pointer = T*;
		using const_pointer = const T*;
		using iterator = T*;
		using const_iterator = const T*;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const iterator>;

		vector(const Allocator& alloc = {}) : alloc(alloc) {}
		vector(size_t count, const T& value, const Allocator& alloc = {}) : alloc(alloc) {
			reserve(count);
			for (size_t i = 0; i < count; ++i)
				this->alloc.template construct<T>(ptr + i, value);
			nStored = count;
		}
		vector(size_t count, const Allocator& alloc = {}) : vector(count, T{}, alloc) {}
		/* Copy constructor */
		vector(const vector& other, const Allocator& alloc = {}) : alloc(alloc) {
			reserve(other.size());
			for (size_t i = 0; i < other.size(); ++i)
				this->alloc.template construct<T>(ptr + i, other[i]);
			nStored = other.size();
		}
		template <class InputIt>
		vector(InputIt first, InputIt last, const Allocator& alloc = {}) : alloc(alloc) {
			reserve(last - first);
			size_t i = 0;
			for (InputIt iter = first; iter != last; ++iter, ++i)
				this->alloc.template construct<T>(ptr + i, *iter);
			nStored = nAlloc;
		}
		/* Move constructor */
		vector(vector&& other) : alloc(other.alloc) {
			nStored = other.nStored;
			nAlloc = other.nAlloc;
			ptr = other.ptr;

			other.nStored = other.nAlloc = 0;
			other.ptr = nullptr;
		}
		vector(vector&& other, const Allocator& alloc) {
			if (alloc == other.alloc) {
				ptr = other.ptr;
				nAlloc = other.nAlloc;
				nStored = other.nStored;

				other.ptr = nullptr;
				other.nAlloc = other.nStored = 0;
			}
			else {
				reserve(other.size());
				for (size_t i = 0; i < other.size(); ++i)
					alloc.template construct<T>(ptr + i, std::move(other[i]));
				nStored = other.size();
			}
		}
		vector(std::initializer_list<T> init, const Allocator& alloc = {})
			: vector(init.begin(), init.end(), alloc) {}

		vector& operator=(const vector& other) {
			if (this == &other)
				return *this;

			clear();
			reserve(other.size());
			for (size_t i = 0; i < other.size(); ++i)
				alloc.template construct<T>(ptr + i, other[i]);
			nStored = other.size();

			return *this;
		}

		vector& operator=(vector&& other) {
			if (this == &other)
				return *this;

			if (alloc == other.alloc) {
				std::swap(ptr, other.ptr);
				std::swap(nAlloc, other.nAlloc);
				std::swap(nStored, other.nStored);
			}
			else {
				clear();
				reserve(other.size());
				for (size_t i = 0; i < other.size(); ++i)
					alloc.template construct<T>(ptr + i, std::move(other[i]));
				nStored = other.size();
			}

			return *this;
		}
		vector& operator=(std::initializer_list<T>& init) {
			reserve(init.size());
			clear();
			iterator iter = begin();
			for (const auto& value : init) {
				*iter = value;
				++iter;
			}
			return *this;
		}

		void assign(size_type count, const T& value) {
			clear();
			reserve(count);
			for (size_t i = 0; i < count; ++i)
				push_back(value);
		}
		template <class InputIt>
		void assign(InputIt first, InputIt last) {
			clear();
			reserve(last - first);
			for (InputIt iter = first; iter != last; ++iter)
				push_back(*iter);
		}
		void assign(std::initializer_list<T>& init) { assign(init.begin(), init.end()); }

		~vector() {
			clear();
			alloc.deallocate_object(ptr, nAlloc);
		}

		KRR_CALLABLE
			iterator begin() { return ptr; }
		KRR_CALLABLE
			iterator end() { return ptr + nStored; }
		KRR_CALLABLE
			const_iterator begin() const { return ptr; }
		KRR_CALLABLE
			const_iterator end() const { return ptr + nStored; }
		KRR_CALLABLE
			const_iterator cbegin() const { return ptr; }
		KRR_CALLABLE
			const_iterator cend() const { return ptr + nStored; }

		KRR_CALLABLE
			reverse_iterator rbegin() { return reverse_iterator(end()); }
		KRR_CALLABLE
			reverse_iterator rend() { return reverse_iterator(begin()); }
		KRR_CALLABLE
			const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
		KRR_CALLABLE
			const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

		allocator_type get_allocator() const { return alloc; }
		KRR_CALLABLE
			size_t size() const { return nStored; }
		KRR_CALLABLE
			bool empty() const { return size() == 0; }
		KRR_CALLABLE
			size_t max_size() const { return (size_t)-1; }
		KRR_CALLABLE
			size_t capacity() const { return nAlloc; }
		void reserve(size_t n) {
			if (nAlloc >= n)
				return;

			T* ra = alloc.template allocate_object<T>(n);
			for (int i = 0; i < nStored; ++i) {
				alloc.template construct<T>(ra + i, std::move(begin()[i]));
				alloc.destroy(begin() + i);
			}

			alloc.deallocate_object(ptr, nAlloc);
			nAlloc = n;
			ptr = ra;
		}
		// TODO: shrink_to_fit

		KRR_CALLABLE
			reference operator[](size_type index) {
			DCHECK_LT(index, size());
			return ptr[index];
		}
		KRR_CALLABLE
			const_reference operator[](size_type index) const {
			DCHECK_LT(index, size());
			return ptr[index];
		}
		KRR_CALLABLE
			reference front() { return ptr[0]; }
		KRR_CALLABLE
			const_reference front() const { return ptr[0]; }
		KRR_CALLABLE
			reference back() { return ptr[nStored - 1]; }
		KRR_CALLABLE
			const_reference back() const { return ptr[nStored - 1]; }
		KRR_CALLABLE
			pointer data() { return ptr; }
		KRR_CALLABLE
			const_pointer data() const { return ptr; }

		void clear() {
			for (int i = 0; i < nStored; ++i)
				alloc.destroy(&ptr[i]);
			nStored = 0;
		}

		template <class... Args>
		void emplace_back(Args &&...args) {
			if (nAlloc == nStored)
				reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

			alloc.construct(ptr + nStored, std::forward<Args>(args)...);
			++nStored;
		}

		void push_back(const T& value) {
			if (nAlloc == nStored)
				reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

			alloc.construct(ptr + nStored, value);
			++nStored;
		}
		void push_back(T&& value) {
			if (nAlloc == nStored)
				reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

			alloc.construct(ptr + nStored, std::move(value));
			++nStored;
		}
		void pop_back() {
			DCHECK(!empty());
			alloc.destroy(ptr + nStored - 1);
			--nStored;
		}

		void resize(size_type n) {
			if (n < size()) {
				for (size_t i = n; i < size(); ++i)
					alloc.destroy(ptr + i);
				if (n == 0) {
					alloc.deallocate_object(ptr, nAlloc);
					ptr = nullptr;
					nAlloc = 0;
				}
			}
			else if (n > size()) {
				reserve(n);
				for (size_t i = nStored; i < n; ++i)
					alloc.construct(ptr + i);
			}
			nStored = n;
		}

		void swap(vector& other) {
			CHECK(alloc == other.alloc);  // TODO: handle this
			std::swap(ptr, other.ptr);
			std::swap(nAlloc, other.nAlloc);
			std::swap(nStored, other.nStored);
		}

	private:
		T* ptr = nullptr;
		size_t nAlloc = 0, nStored = 0;
		Allocator alloc;
	};

	template <typename... Ts> struct tuple;
	template <> struct tuple<> {
		template <size_t> using type = void;
	};

	template <typename T, typename... Ts> struct tuple<T, Ts...> : tuple<Ts...> {
		using Base = tuple<Ts...>;

		tuple()							= default;
		tuple(const tuple &)			= default;
		tuple(tuple &&)					= default;
		tuple &operator=(tuple &&)		= default;
		tuple &operator=(const tuple &) = default;

		tuple(const T &value, const Ts &...values) : Base(values...), value(value) {}

		tuple(T &&value, Ts &&...values) : Base(std::move(values)...), value(std::move(value)) {}

		T value;
	};


	template <typename... Ts> tuple(Ts &&...) -> tuple<std::decay_t<Ts>...>;

	template <size_t I, typename T, typename... Ts> 
	KRR_CALLABLE auto &get(tuple<T, Ts...> &t) {
		if constexpr (I == 0)
			return t.value;
		else
			return get<I - 1>((tuple<Ts...> &) t);
	}

	template <size_t I, typename T, typename... Ts>
	KRR_CALLABLE const auto &get(const tuple<T, Ts...> &t) {
		if constexpr (I == 0)
			return t.value;
		else
			return get<I - 1>((const tuple<Ts...> &) t);
	}

	template <typename Req, typename T, typename... Ts> 
	KRR_CALLABLE auto &get(tuple<T, Ts...> &t) {
		if constexpr (std::is_same_v<Req, T>)
			return t.value;
		else
			return get<Req>((tuple<Ts...> &) t);
	}

	template <typename Req, typename T, typename... Ts>
	KRR_CALLABLE const auto &get(const tuple<T, Ts...> &t) {
		if constexpr (std::is_same_v<Req, T>)
			return t.value;
		else
			return get<Req>((const tuple<Ts...> &) t);
	}

	template <typename T> class optional {
	public:
		using value_type = T;

		optional() = default;
		KRR_CALLABLE optional(const T &v) : set(true) { new (ptr()) T(v); }
		KRR_CALLABLE optional(T &&v) : set(true) { new (ptr()) T(std::move(v)); }
		KRR_CALLABLE optional(const optional &v) : set(v.has_value()) {
			if (v.has_value()) new (ptr()) T(v.value());
		}
		KRR_CALLABLE optional(optional &&v) : set(v.has_value()) {
			if (v.has_value()) {
				new (ptr()) T(std::move(v.value()));
				v.reset();
			}
		}

		KRR_CALLABLE optional &operator=(const T &v) {
			reset();
			new (ptr()) T(v);
			set = true;
			return *this;
		}
		KRR_CALLABLE optional &operator=(T &&v) {
			reset();
			new (ptr()) T(std::move(v));
			set = true;
			return *this;
		}
		KRR_CALLABLE optional &operator=(const optional &v) {
			reset();
			if (v.has_value()) {
				new (ptr()) T(v.value());
				set = true;
			}
			return *this;
		}
		KRR_CALLABLE optional &operator=(optional &&v) {
			reset();
			if (v.has_value()) {
				new (ptr()) T(std::move(v.value()));
				set = true;
				v.reset();
			}
			return *this;
		}

		KRR_CALLABLE ~optional() { reset(); }

		KRR_CALLABLE explicit operator bool() const { return set; }

		KRR_CALLABLE T value_or(const T &alt) const { return set ? value() : alt; }

		KRR_CALLABLE T *operator->() { return &value(); }
		KRR_CALLABLE const T *operator->() const { return &value(); }
		KRR_CALLABLE T &operator*() { return value(); }
		KRR_CALLABLE const T &operator*() const { return value(); }
		KRR_CALLABLE T &value() {
			CHECK(set);
			return *ptr();
		}
		KRR_CALLABLE const T &value() const {
			CHECK(set);
			return *ptr();
		}

		KRR_CALLABLE void reset() {
			if (set) {
				value().~T();
				set = false;
			}
		}

		KRR_CALLABLE bool has_value() const { return set; }

	private:
#ifdef __NVCC__
		// Work-around NVCC bug
		KRR_CALLABLE
		T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
		KRR_CALLABLE
		const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
#else
		KRR_CALLABLE
		T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
		KRR_CALLABLE
		const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
#endif

		std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
		bool set = false;
	};
} // namespace gpu end
NAMESPACE_END(krr)
