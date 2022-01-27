#pragma once

#include "math/vec.h"
#include "math/quat.h"

namespace krr
{
	namespace math
	{
		using namespace polymorphic;
		/// 2D Linear Transform (2x2 Matrix)

		template <typename T>
		struct mat2_t
		{
			using vec_t = T;
			// using Scalar = typename T::scalar_t;
			// using vec_t = T;
			using scalar_t = typename T::scalar_t;

			/*! default matrix constructor */
			inline mat2_t() = default;
			inline __both__ mat2_t(const mat2_t &other)
			{
				vx = other.vx;
				vy = other.vy;
			}
			inline __both__ mat2_t &operator=(const mat2_t &other)
			{
				vx = other.vx;
				vy = other.vy;
				return *this;
			}

			template <typename L1>
			inline __both__ mat2_t(const mat2_t<L1> &s) : vx(s.vx), vy(s.vy) {}

			/*! matrix construction from column vectors */
			inline __both__ mat2_t(const vec_t &vx, const vec_t &vy)
				: vx(vx), vy(vy) {}

			/*! matrix construction from row mayor data */
			inline __both__ mat2_t(const scalar_t &m00, const scalar_t &m01,
								   const scalar_t &m10, const scalar_t &m11)
				: vx(m00, m10), vy(m01, m11) {}

			/*! compute the determinant of the matrix */
			inline __both__ const scalar_t det() const { return vx.x * vy.y - vx.y * vy.x; }

			/*! compute adjoint matrix */
			inline __both__ const mat2_t adjoint() const { return mat2_t(vy.y, -vy.x, -vx.y, vx.x); }

			/*! compute inverse matrix */
			inline __both__ const mat2_t inverse() const { return adjoint() / det(); }

			/*! compute transposed matrix */
			inline __both__ const mat2_t transposed() const { return mat2_t(vx.x, vx.y, vy.x, vy.y); }

			/*! returns first row of matrix */
			inline const vec_t row0() const { return vec_t(vx.x, vy.x); }

			/*! returns second row of matrix */
			inline const vec_t row1() const { return vec_t(vx.y, vy.y); }

			/// Constants

			inline __both__ mat2_t(ZeroTy) : vx(ZeroTy()), vy(ZeroTy()) {}
			inline __both__ mat2_t(OneTy) : vx(OneTy(), ZeroTy()), vy(ZeroTy(), OneTy()) {}

			/*! return matrix for scaling */
			static inline mat2_t scale(const vec_t &s)
			{
				return mat2_t(s.x, 0,
							  0, s.y);
			}

			/*! return matrix for rotation */
			static inline mat2_t rotate(const scalar_t &r)
			{
				scalar_t s = sin(r), c = cos(r);
				return mat2_t(c, -s,
							  s, c);
			}

			/*! return closest orthogonal matrix (i.e. a general rotation including reflection) */
			mat2_t orthogonal() const
			{
				mat2_t m = *this;

				// mirrored?
				scalar_t mirror{scalar_t(OneTy())};
				if (m.det() < scalar_t(ZeroTy()))
				{
					m.vx = -m.vx;
					mirror = -mirror;
				}

				// rotation
				for (int i = 0; i < 99; i++)
				{
					const mat2_t m_next = 0.5 * (m + m.transposed().inverse());
					const mat2_t d = m_next - m;
					m = m_next;
					// norm^2 of difference small enough?
					if (max(dot(d.vx, d.vx), dot(d.vy, d.vy)) < 1e-8)
						break;
				}

				// rotation * mirror_x
				return mat2_t(mirror * m.vx, m.vy);
			}

		public:
			/*! the column vectors of the matrix */
			vec_t vx, vy;
		};

		// Unary Operators

		template <typename T>
		__both__ inline mat2_t<T> operator-(const mat2_t<T> &a) { return mat2_t<T>(-a.vx, -a.vy); }
		template <typename T>
		__both__ inline mat2_t<T> operator+(const mat2_t<T> &a) { return mat2_t<T>(+a.vx, +a.vy); }
		template <typename T>
		__both__ inline mat2_t<T> rcp(const mat2_t<T> &a) { return a.inverse(); }

		// Binary Operators

		template <typename T>
		inline mat2_t<T> operator+(const mat2_t<T> &a, const mat2_t<T> &b) { return mat2_t<T>(a.vx + b.vx, a.vy + b.vy); }
		template <typename T>
		inline mat2_t<T> operator-(const mat2_t<T> &a, const mat2_t<T> &b) { return mat2_t<T>(a.vx - b.vx, a.vy - b.vy); }

		template <typename T>
		inline mat2_t<T> operator*(const typename T::scalar_t &a, const mat2_t<T> &b) { return mat2_t<T>(a * b.vx, a * b.vy); }
		template <typename T>
		inline T operator*(const mat2_t<T> &a, const T &b) { return b.x * a.vx + b.y * a.vy; }
		template <typename T>
		inline mat2_t<T> operator*(const mat2_t<T> &a, const mat2_t<T> &b) { return mat2_t<T>(a * b.vx, a * b.vy); }

		template <typename T>
		inline mat2_t<T> operator/(const mat2_t<T> &a, const typename T::scalar_t &b) { return mat2_t<T>(a.vx / b, a.vy / b); }
		template <typename T>
		inline mat2_t<T> operator/(const mat2_t<T> &a, const mat2_t<T> &b) { return a * rcp(b); }

		template <typename T>
		inline mat2_t<T> &operator*=(mat2_t<T> &a, const mat2_t<T> &b) { return a = a * b; }
		template <typename T>
		inline mat2_t<T> &operator/=(mat2_t<T> &a, const mat2_t<T> &b) { return a = a / b; }

		/// Comparison Operators

		template <typename T>
		inline bool operator==(const mat2_t<T> &a, const mat2_t<T> &b) { return a.vx == b.vx && a.vy == b.vy; }
		template <typename T>
		inline bool operator!=(const mat2_t<T> &a, const mat2_t<T> &b) { return a.vx != b.vx || a.vy != b.vy; }

		/// Output Operators

		template <typename T>
		static std::ostream &operator<<(std::ostream &cout, const mat2_t<T> &m)
		{
			return cout << "{ vx = " << m.vx << ", vy = " << m.vy << "}";
		}

		/// 3D Linear Transform (3x3 Matrix)

		template <typename T>
		struct mat3_t
		{
			// using vec_t = T;
			using scalar_t = typename T::scalar_t;
			using vec_t = T;
			// using scalar_t = typename T::scalar_t;

			/*! default matrix constructor */
			// inline mat3_t           ( ) = default;
			inline __both__ mat3_t()
				: vx(OneTy(), ZeroTy(), ZeroTy()),
				  vy(ZeroTy(), OneTy(), ZeroTy()),
				  vz(ZeroTy(), ZeroTy(), OneTy())
			{
			}

			inline // __both__
				mat3_t(const mat3_t &other) = default;
			inline __both__ mat3_t &operator=(const mat3_t &other)
			{
				vx = other.vx;
				vy = other.vy;
				vz = other.vz;
				return *this;
			}

			template <typename L1>
			inline __both__ mat3_t(const mat3_t<L1> &s) : vx(s.vx), vy(s.vy), vz(s.vz) {}

			/*! matrix construction from column vectors */
			inline __both__ mat3_t(const vec_t &vx, const vec_t &vy, const vec_t &vz)
				: vx(vx), vy(vy), vz(vz) {}

			/*! construction from quaternion */
			inline __both__ mat3_t(const quat_t<scalar_t> &q)
				: vx((q.r * q.r + q.i * q.i - q.j * q.j - q.k * q.k), 2.0f * (q.i * q.j + q.r * q.k), 2.0f * (q.i * q.k - q.r * q.j)), vy(2.0f * (q.i * q.j - q.r * q.k), (q.r * q.r - q.i * q.i + q.j * q.j - q.k * q.k), 2.0f * (q.j * q.k + q.r * q.i)), vz(2.0f * (q.i * q.k + q.r * q.j), 2.0f * (q.j * q.k - q.r * q.i), (q.r * q.r - q.i * q.i - q.j * q.j + q.k * q.k)) {}

			/*! matrix construction from row mayor data */
			inline __both__ mat3_t(const scalar_t &m00, const scalar_t &m01, const scalar_t &m02,
								   const scalar_t &m10, const scalar_t &m11, const scalar_t &m12,
								   const scalar_t &m20, const scalar_t &m21, const scalar_t &m22)
				: vx(m00, m10, m20), vy(m01, m11, m21), vz(m02, m12, m22) {}

			/*! compute the determinant of the matrix */
			inline __both__ const scalar_t det() const { return dot(vx, cross(vy, vz)); }

			/*! compute adjoint matrix */
			inline __both__ const mat3_t adjoint() const { return mat3_t(cross(vy, vz), cross(vz, vx), cross(vx, vy)).transposed(); }

			/*! compute inverse matrix */
			inline __both__ const mat3_t inverse() const { return adjoint() / det(); }

			/*! compute transposed matrix */
			inline __both__ const mat3_t transposed() const { return mat3_t(vx.x, vx.y, vx.z, vy.x, vy.y, vy.z, vz.x, vz.y, vz.z); }

			/*! returns first row of matrix */
			inline __both__ const vec_t row0() const { return vec_t(vx.x, vy.x, vz.x); }

			/*! returns second row of matrix */
			inline __both__ const vec_t row1() const { return vec_t(vx.y, vy.y, vz.y); }

			/*! returns third row of matrix */
			inline __both__ const vec_t row2() const { return vec_t(vx.z, vy.z, vz.z); }

			/// Constants

			inline __both__ mat3_t(const ZeroTy &)
				: vx(ZeroTy()), vy(ZeroTy()), vz(ZeroTy())
			{
			}
			inline __both__ mat3_t(const OneTy &)
				: vx(OneTy(), ZeroTy(), ZeroTy()),
				  vy(ZeroTy(), OneTy(), ZeroTy()),
				  vz(ZeroTy(), ZeroTy(), OneTy())
			{
			}

			/*! return matrix for scaling */
			static inline __both__ mat3_t scale(const vec_t &s)
			{
				return mat3_t(s.x, 0, 0,
							  0, s.y, 0,
							  0, 0, s.z);
			}

			/*! return matrix for rotation around arbitrary axis */
			static inline __both__ mat3_t rotate(const vec_t &_u, const scalar_t &r)
			{
				vec_t u = normalize(_u);
				scalar_t s = sin(r), c = cos(r);
				return mat3_t(u.x * u.x + (1 - u.x * u.x) * c, u.x * u.y * (1 - c) - u.z * s, u.x * u.z * (1 - c) + u.y * s,
							  u.x * u.y * (1 - c) + u.z * s, u.y * u.y + (1 - u.y * u.y) * c, u.y * u.z * (1 - c) - u.x * s,
							  u.x * u.z * (1 - c) - u.y * s, u.y * u.z * (1 - c) + u.x * s, u.z * u.z + (1 - u.z * u.z) * c);
			}

			/*! return quaternion for given rotation matrix */
			static inline __both__ quat_t<scalar_t> rotation(const mat3_t &a)
			{
				scalar_t tr = a.vx.x + a.vy.y + a.vz.z + 1;
				vec_t diag(a.vx.x, a.vy.y, a.vz.z);
				if (tr > 1)
				{
					scalar_t s = sqrt(tr) * 2;
					return quat_t<scalar_t>(.25f * s,
											(a.vz.y - a.vy.z) / s,
											(a.vx.z - a.vz.x) / s,
											(a.vy.x - a.vx.y) / s);
				}
				else if (arg_min(diag) == 0)
				{
					scalar_t s = sqrt(1.f + diag.x - diag.y - diag.z) * 2.f;
					return quat_t<scalar_t>((a.vz.y - a.vy.z) / s,
											.25f * s,
											(a.vx.y - a.vy.x) / s,
											(a.vx.z - a.vz.x) / s);
				}
				else if (arg_min(diag) == 1)
				{
					scalar_t s = sqrt(1.f + diag.y - diag.x - diag.z) * 2.f;
					return quat_t<scalar_t>((a.vx.z - a.vz.x) / s,
											(a.vx.y - a.vy.x) / s,
											.25f * s,
											(a.vy.z - a.vz.y) / s);
				}
				else
				{
					scalar_t s = sqrt(1.f + diag.z - diag.x - diag.y) * 2.f;
					return quat_t<scalar_t>((a.vy.x - a.vx.y) / s,
											(a.vx.z - a.vz.x) / s,
											(a.vy.z - a.vz.y) / s,
											.25f * s);
				}
			}

		public:
			/*! the column vectors of the matrix */
			T vx, vy, vz;
		};

		// Unary Operators

		template <typename T>
		inline __both__ mat3_t<T> operator-(const mat3_t<T> &a) { return mat3_t<T>(-a.vx, -a.vy, -a.vz); }
		template <typename T>
		inline __both__ mat3_t<T> operator+(const mat3_t<T> &a) { return mat3_t<T>(+a.vx, +a.vy, +a.vz); }
		template <typename T>
		inline __both__ mat3_t<T> rcp(const mat3_t<T> &a) { return a.inverse(); }

		/* constructs a coordinate frame form a normalized normal */
		template <typename T>
		inline __both__ mat3_t<T> frame(const T &N)
		{
			const T dx0 = cross(T(OneTy(), ZeroTy(), ZeroTy()), N);
			const T dx1 = cross(T(ZeroTy(), OneTy(), ZeroTy()), N);

			const T dx = normalize(select(dot(dx0, dx0) > dot(dx1, dx1), dx0, dx1));
			const T dy = normalize(cross(N, dx));
			return mat3_t<T>(dx, dy, N);
		}

		/* constructs a coordinate frame from a normal and approximate x-direction */
		template <typename T>
		inline __both__ mat3_t<T> frame(const T &N, const T &dxi)
		{
			if (abs(dot(dxi, N)) > 0.99f)
				return frame(N); // fallback in case N and dxi are very parallel
			const T dx = normalize(cross(dxi, N));
			const T dy = normalize(cross(N, dx));
			return mat3_t<T>(dx, dy, N);
		}

		/* clamps linear space to range -1 to +1 */
		template <typename T>
		inline __both__ mat3_t<T> clamp(const mat3_t<T> &space)
		{
			return mat3_t<T>(clamp(space.vx, T(-1.0f), T(1.0f)),
							 clamp(space.vy, T(-1.0f), T(1.0f)),
							 clamp(space.vz, T(-1.0f), T(1.0f)));
		}

		// Binary Operators

		template <typename T>
		inline __both__ mat3_t<T> operator+(const mat3_t<T> &a, const mat3_t<T> &b) { return mat3_t<T>(a.vx + b.vx, a.vy + b.vy, a.vz + b.vz); }
		template <typename T>
		inline __both__ mat3_t<T> operator-(const mat3_t<T> &a, const mat3_t<T> &b) { return mat3_t<T>(a.vx - b.vx, a.vy - b.vy, a.vz - b.vz); }

		template <typename T>
		inline __both__ mat3_t<T> operator*(const typename T::scalar_t &a, const mat3_t<T> &b) { return mat3_t<T>(a * b.vx, a * b.vy, a * b.vz); }
		template <typename T>
		inline __both__ T operator*(const mat3_t<T> &a, const T &b) { return b.x * a.vx + b.y * a.vy + b.z * a.vz; }
		template <typename T>
		inline __both__ mat3_t<T> operator*(const mat3_t<T> &a, const mat3_t<T> &b) { return mat3_t<T>(a * b.vx, a * b.vy, a * b.vz); }

		template <typename T>
		__both__ inline mat3_t<T> operator/(const mat3_t<T> &a, const typename T::scalar_t &b) { return mat3_t<T>(a.vx / b, a.vy / b, a.vz / b); }

		template <typename T>
		__both__ inline mat3_t<T> operator/(const mat3_t<T> &a, const mat3_t<T> &b) { return a * rcp(b); }

		template <typename T>
		inline mat3_t<T> &operator*=(mat3_t<T> &a, const mat3_t<T> &b) { return a = a * b; }
		template <typename T>
		inline mat3_t<T> &operator/=(mat3_t<T> &a, const mat3_t<T> &b) { return a = a / b; }

		template <typename T>
		inline __both__ T xfmPoint(const mat3_t<T> &s, const T &a) { return madd(T(a.x), s.vx, madd(T(a.y), s.vy, T(a.z * s.vz))); }
		template <typename T>
		inline __both__ T xfmVector(const mat3_t<T> &s, const T &a) { return madd(T(a.x), s.vx, madd(T(a.y), s.vy, T(a.z * s.vz))); }
		template <typename T>
		inline __both__ T xfmNormal(const mat3_t<T> &s, const T &a) { return xfmVector(s.inverse().transposed(), a); }

		/// Comparison Operators

		template <typename T>
		inline bool operator==(const mat3_t<T> &a, const mat3_t<T> &b) { return a.vx == b.vx && a.vy == b.vy && a.vz == b.vz; }
		template <typename T>
		inline bool operator!=(const mat3_t<T> &a, const mat3_t<T> &b) { return a.vx != b.vx || a.vy != b.vy || a.vz != b.vz; }

		/// Output Operators

		template <typename T>
		inline std::ostream &operator<<(std::ostream &cout, const mat3_t<T> &m)
		{
			return cout << "{ vx = " << m.vx << ", vy = " << m.vy << ", vz = " << m.vz << "}";
		}

		using mat2f = mat2_t<vec2f>;
		using mat3f = mat3_t<vec3f>;
		using mat3fa = mat3_t<vec3fa>;

		using linear2f = mat2f;
		using linear3f = mat3f;

	}
}
