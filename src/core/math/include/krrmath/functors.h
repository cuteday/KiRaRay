#pragma once
#include <Eigen/Dense>

#include "common.h"
#include "vector.h"
#include "matrix.h"
#include "constants.h"

KRR_NAMESPACE_BEGIN

#ifdef KRR_DEVICE_CODE
using ::abs;
using ::copysign;
using ::fmod;
using ::max;
using ::min;
using ::isnan;
#else
using std::abs;
using std::copysign;
using std::fmod;
using std::max;
using std::min;
using std::isnan;
#endif

using ::cos;
using ::pow;
using ::sin;
using ::tan;
using ::sinh;
using ::cosh;
using ::tanh;

template <typename T> KRR_CALLABLE auto clamp(T v, T lo, T hi) {
	return std::max(std::min(v, hi), lo);
}

template <typename DerivedV, typename DerivedB>
KRR_CALLABLE auto clamp(const Eigen::MatrixBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.cwiseMin(hi).cwiseMax(lo);
}

template <typename DerivedV, typename DerivedB>
KRR_CALLABLE auto clamp(const Eigen::ArrayBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.min(hi).max(lo);
}

template <typename DerivedV, typename DerivedB>
KRR_CALLABLE auto clamp(const Eigen::EigenBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return clamp(v.eval(), lo, hi);
}

template <typename DerivedA, typename DerivedB, typename DerivedT>
KRR_CALLABLE auto lerp(const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
					   DerivedT t) {
	return (a.eval() * (1 - t) + b.eval() * t).eval();
}

// overload unary opeartors

template <typename DerivedV> KRR_CALLABLE auto normalize(const Eigen::MatrixBase<DerivedV> &v) {
	return v.normalized();
}

template <typename DerivedV> KRR_CALLABLE auto abs(const Eigen::MatrixBase<DerivedV> &v) {
	return v.cwiseAbs();
}

template <typename DerivedV> KRR_CALLABLE auto length(const Eigen::MatrixBase<DerivedV> &v) {
	return v.norm();
}

template <typename DerivedV> KRR_CALLABLE auto squaredLength(const Eigen::MatrixBase<DerivedV> &v) {
	return v.SquaredNorm();
}

template <typename DerivedV> KRR_CALLABLE auto any(const Eigen::DenseBase<DerivedV> &v) {
	return v.any();
}

// overload binary operators

template <typename DerivedA, typename DerivedB>
KRR_CALLABLE auto cross(const Eigen::MatrixBase<DerivedA> &a,
						const Eigen::MatrixBase<DerivedB> &b) {
	return a.cross(b);
}

template <typename DerivedA, typename DerivedB>
KRR_CALLABLE auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.dot(b);
}

template <typename DerivedA, typename DerivedB>
KRR_CALLABLE auto operator/(const Eigen::MatrixBase<DerivedA> &a,
							const Eigen::MatrixBase<DerivedB> &b) {
	return a.cwiseQuotient(b);
}

// power shortcuts
template <typename T> KRR_CALLABLE constexpr T pow1(T x) { return x; }
template <typename T> KRR_CALLABLE constexpr T pow2(T x) { return x * x; }
template <typename T> KRR_CALLABLE constexpr T pow3(T x) { return x * x * x; }
template <typename T> KRR_CALLABLE constexpr T pow4(T x) { return x * x * x * x; }
template <typename T> KRR_CALLABLE constexpr T pow5(T x) { return x * x * x * x * x; }

KRR_CALLABLE float sqrt(const float v) { return sqrtf(v); }

template <typename T> KRR_CALLABLE T mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T) ((result < 0) ? result + b : result);
}

template <typename T> KRR_CALLABLE T safe_sqrt(T value) {
	return sqrt(max((T) 0, value));
}

KRR_CALLABLE float saturate(const float &f) { return min(1.f, max(0.f, f)); }

KRR_CALLABLE float rcp(float f) { return 1.f / f; }

KRR_CALLABLE float logistic(const float x) { return 1 / (1.f + expf(-x)); }

KRR_CALLABLE float csch(const float x) { return 1 / sinh(x); }

KRR_CALLABLE float coth(const float x) { return 1 / tanh(x); }

KRR_CALLABLE float sech(const float x) { return 1 / cosh(x); }

KRR_CALLABLE float radians(const float degree) { return degree * M_PI / 180.f; }

/* space transformations (all in left-handed coordinate) */

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> perspective(T fovy, T aspect, T zNear, T zFar) {
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

	T const tanHalfFovy = tan(fovy / static_cast<T>(2));
	Matrix<T, 4, 4, Options> result{ Matrix<T, 4, 4, Options>::Zero() };
	
	result(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovy);
	result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
	result(2, 2) = -(zFar + zNear) / (zFar - zNear);
	result(2, 3) = -static_cast<T>(1);
	result(3, 2) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
	return result;
}

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> orthogonal(T left, T right, T bottom, T top) {
	Matrix<T, 4, 4, Options> result{ Matrix<T, 4, 4, Options>::Identity() };

	result(0, 0) = static_cast<T>(2) / (right - left);
	result(1, 1) = static_cast<T>(2) / (top - bottom);
	result(2, 2) = -static_cast<T>(1);
	result(3, 0) = -(right + left) / (right - left);
	result(3, 1) = -(top + bottom) / (top - bottom);
	return result;
}

template <typename T, int Options = math::ColMajor>
Matrix<T, 4, 4, Options> look_at(Vector3<T> const &eye, Vector3<T> const &center,
								 Vector3<T> const &up) {
	Vector3<T> const f(normalize(center - eye));
	Vector3<T> const s(normalize(cross(up, f)));
	Vector3<T> const u(cross(f, s));

	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};
	result(0, 0) = s.x;
	result(1, 0) = s.y;
	result(2, 0) = s.z;
	result(0, 1) = u.x;
	result(1, 1) = u.y;
	result(2, 1) = u.z;
	result(0, 2) = f.x;
	result(1, 2) = f.y;
	result(2, 2) = f.z;
	result(3, 0) = -dot(s, eye);
	result(3, 1) = -dot(u, eye);
	result(3, 2) = -dot(f, eye);
	return result;
}

KRR_NAMESPACE_END