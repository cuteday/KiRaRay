#pragma once

#include "math/mat.h"
#include "math/aabb.h"

namespace krr
{
	namespace math
	{

		#define vec_t typename L::vec_t
		#define scalar_t vec_t::scalar_t	// becareful of that recursive definition! 

		// Affine Space
		template <typename L>
		struct transform_t
		{
			// using vec_t L::vec_t;
			// using scalar_t L::vec_t::scalar_t;

			L l;	 /*< linear part of affine space */
			vec_t p; /*< affine part of affine space */

			// Constructors, Assignment, Cast, Copy Operations

			inline __both__
			transform_t()
				: l(OneTy()),
				  p(ZeroTy())
			{
			}

			inline // __both__
				transform_t(const transform_t &other) = default;
			inline __both__ transform_t(const L &other)
			{
				l = other;
				p = vec_t(ZeroTy());
			}
			inline __both__ transform_t &operator=(const transform_t &other)
			{
				l = other.l;
				p = other.p;
				return *this;
			}

			inline __both__ transform_t(const vec_t &vx, const vec_t &vy, const vec_t &vz, const vec_t &p) : l(vx, vy, vz), p(p) {}
			inline __both__ transform_t(const L &l, const vec_t &p) : l(l), p(p) {}

			template <typename L1>
			inline __both__ transform_t(const transform_t<L1> &s) : l(s.l), p(s.p) {}

			// Constants

			inline transform_t(ZeroTy) : l(ZeroTy()), p(ZeroTy()) {}
			inline transform_t(OneTy) : l(OneTy()), p(ZeroTy()) {}

			/*! return matrix for scaling */
			static inline transform_t scale(const vec_t &s) { return L::scale(s); }

			/*! return matrix for translation */
			static inline transform_t translate(const vec_t &p) { return transform_t(OneTy(), p); }

			/*! return matrix for rotation, only in 2D */
			static inline transform_t rotate(const scalar_t &r) { return L::rotate(r); }

			/*! return matrix for rotation around arbitrary point (2D) or axis (3D) */
			static inline transform_t rotate(const vec_t &u, const scalar_t &r) { return L::rotate(u, r); }

			/*! return matrix for rotation around arbitrary axis and point, only in 3D */
			static inline transform_t rotate(const vec_t &p, const vec_t &u, const scalar_t &r) { return translate(+p) * rotate(u, r) * translate(-p); }

			/*! return matrix for looking at given point, only in 3D; right-handed coordinate system */
			static inline transform_t lookat(const vec_t &eye, const vec_t &point, const vec_t &up)
			{
				vec_t Z = normalize(point - eye);
				vec_t U = normalize(cross(Z, up));
				vec_t V = cross(U, Z);
				return transform_t(L(U, V, Z), eye);
			}
		};

		// Unary Operators

		template <typename L>
		inline transform_t<L> operator-(const transform_t<L> &a) { return transform_t<L>(-a.l, -a.p); }
		template <typename L>
		inline transform_t<L> operator+(const transform_t<L> &a) { return transform_t<L>(+a.l, +a.p); }
		template <typename L>
		inline __both__
			transform_t<L>
			rcp(const transform_t<L> &a)
		{
			L il = rcp(a.l);
			return transform_t<L>(il, -(il * a.p));
		}

		// Binary Operators

		template <typename L>
		inline transform_t<L> operator+(const transform_t<L> &a, const transform_t<L> &b) { return transform_t<L>(a.l + b.l, a.p + b.p); }
		template <typename L>
		inline transform_t<L> operator-(const transform_t<L> &a, const transform_t<L> &b) { return transform_t<L>(a.l - b.l, a.p - b.p); }

		template <typename L>
		inline transform_t<L> operator*(const scalar_t &a, const transform_t<L> &b) { return transform_t<L>(a * b.l, a * b.p); }
		template <typename L>
		inline transform_t<L> operator*(const transform_t<L> &a, const transform_t<L> &b) { return transform_t<L>(a.l * b.l, a.l * b.p + a.p); }
		template <typename L>
		inline transform_t<L> operator/(const transform_t<L> &a, const transform_t<L> &b) { return a * rcp(b); }
		template <typename L>
		inline transform_t<L> operator/(const transform_t<L> &a, const scalar_t &b) { return a * rcp(b); }

		template <typename L>
		inline transform_t<L> &operator*=(transform_t<L> &a, const transform_t<L> &b) { return a = a * b; }
		template <typename L>
		inline transform_t<L> &operator*=(transform_t<L> &a, const scalar_t &b) { return a = a * b; }
		template <typename L>
		inline transform_t<L> &operator/=(transform_t<L> &a, const transform_t<L> &b) { return a = a / b; }
		template <typename L>
		inline transform_t<L> &operator/=(transform_t<L> &a, const scalar_t &b) { return a = a / b; }

		template <typename L>
		inline __both__ const vec_t xfmPoint(const transform_t<L> &m, const vec_t &p) { return madd(vec_t(p.x), m.l.vx, madd(vec_t(p.y), m.l.vy, madd(vec_t(p.z), m.l.vz, m.p))); }
		template <typename L>
		inline __both__ const vec_t xfmVector(const transform_t<L> &m, const vec_t &v) { return xfmVector(m.l, v); }
		template <typename L>
		inline __both__ const vec_t xfmNormal(const transform_t<L> &m, const vec_t &n) { return xfmNormal(m.l, n); }

		/// Comparison Operators

		template <typename L>
		inline bool operator==(const transform_t<L> &a, const transform_t<L> &b) { return a.l == b.l && a.p == b.p; }
		template <typename L>
		inline bool operator!=(const transform_t<L> &a, const transform_t<L> &b) { return a.l != b.l || a.p != b.p; }

		// Output Operators

		template <typename L>
		inline std::ostream &operator<<(std::ostream &cout, const transform_t<L> &m)
		{
			return cout << "{ l = " << m.l << ", p = " << m.p << " }";
		}

		// Type Aliases

		using transform2f      = transform_t<mat2f>;
		using transform3f      = transform_t<mat3f>;
		using transform3fa     = transform_t<mat3fa>;
		using orthonormal3f = transform_t<quat3f>;

		using affine2f = transform2f;
		using affine3f = transform3f;

		/*! Template Specialization for 2D: return matrix for rotation around point (rotation around arbitrarty vector is not meaningful in 2D) */
		template<>
		inline transform2f transform2f::rotate(const vec2f& p, const float& r)
		{ return translate(+p) * transform2f(mat2f::rotate(r)) * translate(-p); }

		#undef vec_t
		#undef scalar_t

		    inline __both__ box3f xfmBounds(const affine3f &xfm,
		                                    const box3f &box)
		    {
		      box3f dst;
		      const vec3f lo = box.lower;
		      const vec3f hi = box.upper;
		      dst.extend(xfmPoint(xfm,vec3f(lo.x,lo.y,lo.z)));
		      dst.extend(xfmPoint(xfm,vec3f(lo.x,lo.y,hi.z)));
		      dst.extend(xfmPoint(xfm,vec3f(lo.x,hi.y,lo.z)));
		      dst.extend(xfmPoint(xfm,vec3f(lo.x,hi.y,hi.z)));
		      dst.extend(xfmPoint(xfm,vec3f(hi.x,lo.y,lo.z)));
		      dst.extend(xfmPoint(xfm,vec3f(hi.x,lo.y,hi.z)));
		      dst.extend(xfmPoint(xfm,vec3f(hi.x,hi.y,lo.z)));
		      dst.extend(xfmPoint(xfm,vec3f(hi.x,hi.y,hi.z)));
		      return dst;
		    }
	}
}
