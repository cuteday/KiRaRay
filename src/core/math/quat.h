#pragma once

#include "math/vec.h"

namespace krr
{
	namespace math
	{
		//using namespace polymorphic;
		//using polymorphic::rsqrt;

		// Quaternion Struct
		template <typename T>
		struct quat_t
		{
			typedef vec_t<T, 3> vec;

			/// Construction

			__both__ quat_t(void) {}
			__both__ quat_t(const quat_t &other)
			{
				r = other.r;
				i = other.i;
				j = other.j;
				k = other.k;
			}
			__both__ quat_t &operator=(const quat_t &other)
			{
				r = other.r;
				i = other.i;
				j = other.j;
				k = other.k;
				return *this;
			}

			__both__ quat_t(const T &r) : r(r), i(zero), j(zero), k(zero) {}
			__both__ explicit quat_t(const vec &v) : r(zero), i(v.x), j(v.y), k(v.z) {}
			__both__ quat_t(const T &r, const T &i, const T &j, const T &k) : r(r), i(i), j(j), k(k) {}
			__both__ quat_t(const T &r, const vec &v) : r(r), i(v.x), j(v.y), k(v.z) {}

			__inline quat_t(const vec &vx, const vec &vy, const vec &vz);
			__inline quat_t(const T &yaw, const T &pitch, const T &roll);

			/// Constants

#ifdef __NVCC__
			__both__ quat_t(const ZeroTy &) : r(zero), i(zero), j(zero), k(zero)
			{
			}
			__both__ quat_t(const OneTy &) : r(one), i(zero), j(zero), k(zero) {}
#else
			__both__ quat_t(ZeroTy) : r(zero), i(zero), j(zero), k(zero)
			{
			}
			__both__ quat_t(OneTy) : r(one), i(zero), j(zero), k(zero) {}
#endif

			/*! return quaternion for rotation around arbitrary axis */
			static __both__ quat_t rotate(const vec &u, const T &r)
			{
				return quat_t<T>(cos(T(0.5) * r), sin(T(0.5) * r) * normalize(u));
			}

			/*! returns the rotation axis of the quaternion as a vector */
			__both__ const vec v() const { return vec(i, j, k); }

		public:
			T r, i, j, k;
		};

		template <typename T>
		__both__ quat_t<T> operator*(const T &a, const quat_t<T> &b) { return quat_t<T>(a * b.r, a * b.i, a * b.j, a * b.k); }
		template <typename T>
		__both__ quat_t<T> operator*(const quat_t<T> &a, const T &b) { return quat_t<T>(a.r * b, a.i * b, a.j * b, a.k * b); }

		// Unary Operators

		template <typename T>
		__both__ quat_t<T> operator+(const quat_t<T> &a) { return quat_t<T>(+a.r, +a.i, +a.j, +a.k); }
		template <typename T>
		__both__ quat_t<T> operator-(const quat_t<T> &a) { return quat_t<T>(-a.r, -a.i, -a.j, -a.k); }
		template <typename T>
		__both__ quat_t<T> conj(const quat_t<T> &a) { return quat_t<T>(a.r, -a.i, -a.j, -a.k); }
		template <typename T>
		__both__ T abs(const quat_t<T> &a) { return polymorphic::sqrt(a.r * a.r + a.i * a.i + a.j * a.j + a.k * a.k); }
		template <typename T>
		__both__ quat_t<T> rcp(const quat_t<T> &a) { return conj(a) * rcp(a.r * a.r + a.i * a.i + a.j * a.j + a.k * a.k); }
		template <typename T>
		__both__ quat_t<T> normalize(const quat_t<T> &a) { return a * polymorphic::rsqrt(a.r * a.r + a.i * a.i + a.j * a.j + a.k * a.k); }

		// Binary Operators

		template <typename T>
		__both__ quat_t<T> operator+(const T &a, const quat_t<T> &b) { return quat_t<T>(a + b.r, b.i, b.j, b.k); }
		template <typename T>
		__both__ quat_t<T> operator+(const quat_t<T> &a, const T &b) { return quat_t<T>(a.r + b, a.i, a.j, a.k); }
		template <typename T>
		__both__ quat_t<T> operator+(const quat_t<T> &a, const quat_t<T> &b) { return quat_t<T>(a.r + b.r, a.i + b.i, a.j + b.j, a.k + b.k); }
		template <typename T>
		__both__ quat_t<T> operator-(const T &a, const quat_t<T> &b) { return quat_t<T>(a - b.r, -b.i, -b.j, -b.k); }
		template <typename T>
		__both__ quat_t<T> operator-(const quat_t<T> &a, const T &b) { return quat_t<T>(a.r - b, a.i, a.j, a.k); }
		template <typename T>
		__both__ quat_t<T> operator-(const quat_t<T> &a, const quat_t<T> &b) { return quat_t<T>(a.r - b.r, a.i - b.i, a.j - b.j, a.k - b.k); }

		template <typename T>
		__both__ typename quat_t<T>::vec operator*(const quat_t<T> &a, const typename quat_t<T>::vec &b) { return (a * quat_t<T>(b) * conj(a)).v(); }
		template <typename T>
		__both__ quat_t<T> operator*(const quat_t<T> &a, const quat_t<T> &b)
		{
			return quat_t<T>(a.r * b.r - a.i * b.i - a.j * b.j - a.k * b.k,
							 a.r * b.i + a.i * b.r + a.j * b.k - a.k * b.j,
							 a.r * b.j - a.i * b.k + a.j * b.r + a.k * b.i,
							 a.r * b.k + a.i * b.j - a.j * b.i + a.k * b.r);
		}
		template <typename T>
		__both__ quat_t<T> operator/(const T &a, const quat_t<T> &b) { return a * rcp(b); }
		template <typename T>
		__both__ quat_t<T> operator/(const quat_t<T> &a, const T &b) { return a * rcp(b); }
		template <typename T>
		__both__ quat_t<T> operator/(const quat_t<T> &a, const quat_t<T> &b) { return a * rcp(b); }

		template <typename T>
		__both__ quat_t<T> &operator+=(quat_t<T> &a, const T &b) { return a = a + b; }
		template <typename T>
		__both__ quat_t<T> &operator+=(quat_t<T> &a, const quat_t<T> &b) { return a = a + b; }
		template <typename T>
		__both__ quat_t<T> &operator-=(quat_t<T> &a, const T &b) { return a = a - b; }
		template <typename T>
		__both__ quat_t<T> &operator-=(quat_t<T> &a, const quat_t<T> &b) { return a = a - b; }
		template <typename T>
		__both__ quat_t<T> &operator*=(quat_t<T> &a, const T &b) { return a = a * b; }
		template <typename T>
		__both__ quat_t<T> &operator*=(quat_t<T> &a, const quat_t<T> &b) { return a = a * b; }
		template <typename T>
		__both__ quat_t<T> &operator/=(quat_t<T> &a, const T &b) { return a = a * rcp(b); }
		template <typename T>
		__both__ quat_t<T> &operator/=(quat_t<T> &a, const quat_t<T> &b) { return a = a * rcp(b); }

		template <typename T>
		__both__ typename quat_t<T>::vec
		xfmPoint(const quat_t<T> &a,
				 const typename quat_t<T>::vec &b)
		{
			return (a * quat_t<T>(b) * conj(a)).v();
		}

		template <typename T>
		__both__ typename quat_t<T>::vec
		xfmQuaternion(const quat_t<T> &a,
					  const typename quat_t<T>::vec &b)
		{
			return (a * quat_t<T>(b) * conj(a)).v();
		}

		template <typename T>
		__both__ typename quat_t<T>::vec
		xfmNormal(const quat_t<T> &a,
				  const typename quat_t<T>::vec &b)
		{
			return (a * quat_t<T>(b) * conj(a)).v();
		}

		/// Comparison Operators

		template <typename T>
		__both__ bool operator==(const quat_t<T> &a, const quat_t<T> &b) { return a.r == b.r && a.i == b.i && a.j == b.j && a.k == b.k; }

		template <typename T>
		__both__ bool operator!=(const quat_t<T> &a, const quat_t<T> &b) { return a.r != b.r || a.i != b.i || a.j != b.j || a.k != b.k; }

		/// Orientation Functions

		template <typename T>
		quat_t<T>::quat_t(const typename quat_t<T>::vec &vx,
						  const typename quat_t<T>::vec &vy,
						  const typename quat_t<T>::vec &vz)
		{
			if (vx.x + vy.y + vz.z >= T(zero))
			{
				const T t = T(one) + (vx.x + vy.y + vz.z);
				const T s = rsqrt(t) * T(0.5f);
				r = t * s;
				i = (vy.z - vz.y) * s;
				j = (vz.x - vx.z) * s;
				k = (vx.y - vy.x) * s;
			}
			else if (vx.x >= max(vy.y, vz.z))
			{
				const T t = (T(one) + vx.x) - (vy.y + vz.z);
				const T s = rsqrt(t) * T(0.5f);
				r = (vy.z - vz.y) * s;
				i = t * s;
				j = (vx.y + vy.x) * s;
				k = (vz.x + vx.z) * s;
			}
			else if (vy.y >= vz.z) // if ( vy.y >= max(vz.z, vx.x) )
			{
				const T t = (T(one) + vy.y) - (vz.z + vx.x);
				const T s = rsqrt(t) * T(0.5f);
				r = (vz.x - vx.z) * s;
				i = (vx.y + vy.x) * s;
				j = t * s;
				k = (vy.z + vz.y) * s;
			}
			else //if ( vz.z >= max(vy.y, vx.x) )
			{
				const T t = (T(one) + vz.z) - (vx.x + vy.y);
				const T s = rsqrt(t) * T(0.5f);
				r = (vx.y - vy.x) * s;
				i = (vz.x + vx.z) * s;
				j = (vy.z + vz.y) * s;
				k = t * s;
			}
		}

		template <typename T>
		quat_t<T>::quat_t(const T &yaw, const T &pitch, const T &roll)
		{
			const T cya = cos(yaw * T(0.5f));
			const T cpi = cos(pitch * T(0.5f));
			const T cro = cos(roll * T(0.5f));
			const T sya = sin(yaw * T(0.5f));
			const T spi = sin(pitch * T(0.5f));
			const T sro = sin(roll * T(0.5f));
			r = cro * cya * cpi + sro * sya * spi;
			i = cro * cya * spi + sro * sya * cpi;
			j = cro * sya * cpi - sro * cya * spi;
			k = sro * cya * cpi - cro * sya * spi;
		}

		/// Output Operators

		template <typename T>
		static std::ostream &operator<<(std::ostream &cout, const quat_t<T> &q)
		{
			return cout << "{ r = " << q.r << ", i = " << q.i << ", j = " << q.j << ", k = " << q.k << " }";
		}

		/*! default template instantiations */
		typedef quat_t<float> quat;
		typedef quat_t<float> quat3f;
		typedef quat_t<double> quat3d;

	}
}
