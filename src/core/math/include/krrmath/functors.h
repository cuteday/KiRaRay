#pragma once
#include <Eigen/Dense>

#include "common.h"
#include "vector.h"
#include "matrix.h"
#include "constants.h"

NAMESPACE_BEGIN(krr)

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
using ::atan2;
using ::fma;

template <typename T> KRR_CALLABLE void swap(T &a, T &b) {
	T tmp = std::move(a);
	a	  = std::move(b);
	b	  = std::move(tmp);
}

template <typename T> KRR_CALLABLE auto clamp(T v, T lo, T hi) {
	return std::max(std::min(v, hi), lo);
}

template <typename T, typename U, typename V>
KRR_CALLABLE constexpr std::enable_if_t<std::is_fundamental_v<T>, T> clamp(T val, U low, V high) {
	if (val < low) return T(low);
	else if (val > high) return T(high);
	else return val;
}

template <typename DerivedV, typename DerivedB>
KRR_CALLABLE auto clamp(const Eigen::MatrixBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.cwiseMin(hi).cwiseMax(lo);
}

template <typename DerivedV, typename DerivedB>
KRR_CALLABLE auto clamp(const Eigen::ArrayBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.min(hi).max(lo);
}

template <typename T>
KRR_CALLABLE auto safediv(const Eigen::EigenBase<T> &v, const Eigen::ArrayBase<T> &divisor) {
	return v.binaryExpr(divisor, [](auto x, auto y) { return y == 0 ? 0 : x / y; });
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

template <typename T> KRR_CALLABLE T smooth_step(T x, T a, T b) {
	if (a == b) return (x < a) ? 0 : 1;
	T t = clamp((x - a) / (b - a), 0, 1);
	return t * t * (3 - 2 * t);
}

KRR_CALLABLE float saturate(const float &f) { return min(1.f, max(0.f, f)); }

KRR_CALLABLE float rcp(float f) { return 1.f / f; }

KRR_CALLABLE float logistic(const float x) { return 1 / (1.f + expf(-x)); }

KRR_CALLABLE float csch(const float x) { return 1 / sinh(x); }

KRR_CALLABLE float coth(const float x) { return 1 / tanh(x); }

KRR_CALLABLE float sech(const float x) { return 1 / cosh(x); }

KRR_CALLABLE float radians(const float degree) { return degree * M_PI / 180.f; }

template <typename T> KRR_CALLABLE 
T lerp(T x, T y, T weight) { return (1.f - weight) * x + weight * y; }

NAMESPACE_END(krr)