#pragma once

#include "math/vec.h"

namespace krr {
	namespace math {

		template<typename T>
		struct interval {
			typedef T scalar_t;
			inline __both__ interval()
				: lower(krr::math::empty_bounds_lower<T>()),
				upper(krr::math::empty_bounds_upper<T>())
			{}
			inline __both__ interval(T begin, T end) : begin(begin), end(end) {}

			union {
				T begin;
				T lower;
				T lo;
			};
			union {
				T end;
				T upper;
				T hi;
			};

			inline __both__ bool contains(const T& t) const { return t >= lower && t <= upper; }
			inline __both__ bool is_empty() const { return begin > end; }
			inline __both__ T center() const { return (begin + end) / 2; }
			inline __both__ T extent() const { return end - begin; }
			inline __both__ T span() const { return end - begin; }
			inline __both__ T diagonal() const { return end - begin; }
			inline __both__ interval<T>& extend(const T& t)
			{
				lower = min(lower, t); upper = max(upper, t); return *this;
			}
			inline __both__ interval<T>& extend(const interval<T>& t)
			{
				lower = min(lower, t.lower); upper = max(upper, t.upper); return *this;
			}
			inline __both__ interval<T> including(const T& t)
			{
				return interval<T>(min(lower, t), max(upper, t));
			}

			static inline __both__ interval<T> positive()
			{
				return interval<T>(0.f, krr::math::open_range_upper<T>());
			}
		};

		template<typename T>
		inline __both__ std::ostream& operator<<(std::ostream& o, const interval<T>& b)
		{
#ifndef __NVCC__
			o << "[" << b.lower << ":" << b.upper << "]";
#endif
			return o;
		}

		template<typename T>
		inline __both__ interval<T> build_interval(const T& a, const T& b)
		{
			return interval<T>(min(a, b), max(a, b));
		}

		template<typename T>
		inline __both__ interval<T> intersect(const interval<T>& a, const interval<T>& b)
		{
			return interval<T>(max(a.lower, b.lower), min(a.upper, b.upper));
		}

		template<typename T>
		inline __both__ interval<T> operator-(const interval<T>& a, const T& b)
		{
			return interval<T>(a.lower - b, a.upper - b);
		}

		template<typename T>
		inline __both__ interval<T> operator*(const interval<T>& a, const T& b)
		{
			return build_interval<T>(a.lower * b, a.upper * b);
		}

		template<typename T>
		inline __both__ bool operator==(const interval<T>& a, const interval<T>& b)
		{
			return a.lower == b.lower && a.upper == b.upper;
		}

		template<typename T>
		inline __both__ bool operator!=(const interval<T>& a, const interval<T>& b)
		{
			return !(a == b);
		}



		template<typename T>
		struct aabb_t {
			typedef T vec_t;
			typedef typename T::scalar_t scalar_t;
			enum { dims = T::dims };

			inline __both__ aabb_t()
				: lower(krr::math::empty_bounds_lower<typename T::scalar_t>()),
				upper(krr::math::empty_bounds_upper<typename T::scalar_t>())
			{}

			// /*! construct a new, origin-oriented box of given size */
			// explicit inline __both__ aabb_t(const vec_t &box_size)
			//   : lower(vec_t(0)),
			//     upper(box_size)
			// {}
			/*! construct a new box around a single point */
			explicit inline __both__ aabb_t(const vec_t& v)
				: lower(v),
				upper(v)
			{}

			/*! construct a new, origin-oriented box of given size */
			inline __both__ aabb_t(const vec_t& lo, const vec_t& hi)
				: lower(lo),
				upper(hi)
			{}

			/*! returns new box including both ourselves _and_ the given point */
			inline __both__ aabb_t including(const vec_t& other) const
			{
				return aabb_t(min(lower, other), max(upper, other));
			}
			/*! returns new box including both ourselves _and_ the given point */
			inline __both__ aabb_t including(const aabb_t& other) const
			{
				return aabb_t(min(lower, other.lower), max(upper, other.upper));
			}


			/*! returns new box including both ourselves _and_ the given point */
			inline __both__ aabb_t& extend(const vec_t& other)
			{
				lower = min(lower, other); upper = max(upper, other); return *this;
			}
			/*! returns new box including both ourselves _and_ the given point */
			inline __both__ aabb_t& extend(const aabb_t& other)
			{
				lower = min(lower, other.lower); upper = max(upper, other.upper); return *this;
			}


			/*! get the d-th dimensional slab (lo[dim]..hi[dim] */
			inline __both__ interval<scalar_t> get_slab(const uint32_t dim)
			{
				return interval<scalar_t>(lower[dim], upper[dim]);
			}

			inline __both__ bool contains(const vec_t& point) const
			{
				return !(any_less_than(point, lower) || any_greater_than(point, upper));
			}

			inline __both__ bool overlaps(const aabb_t& other) const
			{
				return !(any_less_than(other.upper, lower) || any_greater_than(other.lower, upper));
			}

			inline __both__ vec_t center() const { return (lower + upper) / (typename vec_t::scalar_t)2; }
			inline __both__ vec_t extent() const { return upper - lower; }
			inline __both__ vec_t span()   const { return upper - lower; }
			inline __both__ vec_t size()   const { return upper - lower; }

			inline __both__ typename long_type_of<typename T::scalar_t>::type volume() const
			{
				return krr::math::volume(size());
			}

			inline __both__ bool empty() const { return any_less_than(upper, lower); }

			vec_t lower, upper;
		};

		// default functions

		template<typename T>
		inline __both__ typename long_type_of<T>::type area(const aabb_t<vec_t<T, 2>>& b)
		{
			return area(b.upper - b.lower);
		}

		template<typename T>
		inline __both__ typename long_type_of<T>::type area(const aabb_t<vec_t<T, 3>>& b)
		{
			const vec_t<T, 3> diag = b.upper - b.lower;
			return 2.f * (area(vec_t<T, 2>(diag.x, diag.y)) +
				area(vec_t<T, 2>(diag.y, diag.z)) +
				area(vec_t<T, 2>(diag.z, diag.x)));
		}

		template<typename T>
		inline __both__ typename long_type_of<T>::type volume(const aabb_t<vec_t<T, 3>>& b)
		{
			const vec_t<T, 3> diag = b.upper - b.lower;
			return diag.x * diag.y * diag.z;
		}

		template<typename T>
		inline __both__ std::ostream& operator<<(std::ostream& o, const aabb_t<T>& b)
		{
#ifndef __NVCC__
			o << "[" << b.lower << ":" << b.upper << "]";
#endif
			return o;
		}

		template<typename T>
		inline __both__ aabb_t<T> intersection(const aabb_t<T>& a, const aabb_t<T>& b)
		{
			return aabb_t<T>(max(a.lower, b.lower), min(a.upper, b.upper));
		}

		template<typename T>
		inline __both__ bool operator==(const aabb_t<T>& a, const aabb_t<T>& b)
		{
			return a.lower == b.lower && a.upper == b.upper;
		}

		template<typename T>
		inline __both__ bool operator!=(const aabb_t<T>& a, const aabb_t<T>& b)
		{
			return !(a == b);
		}

	}
#define _define_aabb_types(T,t)                               \
	typedef math::aabb_t<math::vec_t<T,2>> aabb2##t;          \
	typedef math::aabb_t<math::vec_t<T,3>> aabb3##t;          \
	typedef math::aabb_t<math::vec_t<T,4>> aabb4##t;          \
	typedef math::aabb_t<math::vec3a_t<T>> aabb3##t##a;       \

	_define_aabb_types(bool, b);
	_define_aabb_types(int8_t, c);
	_define_aabb_types(int16_t, s);
	_define_aabb_types(int32_t, i);
	_define_aabb_types(int64_t, l);
	_define_aabb_types(uint8_t, uc);
	_define_aabb_types(uint16_t, us);
	_define_aabb_types(uint32_t, ui);
	_define_aabb_types(uint64_t, ul);
	_define_aabb_types(float, f);
	_define_aabb_types(double, d);

#undef _define_aabb_types
}