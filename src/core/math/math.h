#pragma once

#include "common.h"
#include "math/constants.h"
#include "math/vec.h"
#include "math/functor.h"
//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

using Color = arr3f;
using Color3f = arr3f;
using Point = vec3f;
using Point3f = vec3f;
using Point2f = vec2f;
using Aabb3f = Eigen::AlignedBox<float, 3>;
using Aabb = Aabb3f;
using AABB = Aabb;

namespace math {

template <typename T>
KRR_DEVICE_FUNCTION auto clamp(T v, T lo, T hi) {
	return max(min(v, hi), lo);
}

template <typename DerivedV, typename DerivedB>
KRR_DEVICE_FUNCTION auto clamp(const Eigen::MatrixBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.cwiseMin(hi).cwiseMax(lo);
}

template <typename DerivedA, typename DerivedB, typename DerivedT>
KRR_DEVICE_FUNCTION auto lerp(const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b, DerivedT t) {
	return a.eval() * (1 - t) + b.eval() * t;
}

// overload unary opeartors

template <typename DerivedV>
KRR_DEVICE_FUNCTION auto normalize(const Eigen::MatrixBase<DerivedV> &v) {
	return v.normalized();
}

template <typename DerivedV>
KRR_DEVICE_FUNCTION auto abs(const Eigen::MatrixBase<DerivedV> &v) {
	return v.cwiseAbs();
}

template <typename DerivedV>
KRR_DEVICE_FUNCTION auto length(const Eigen::MatrixBase<DerivedV> &v) {
	return v.norm();
}

template <typename DerivedV>
KRR_DEVICE_FUNCTION auto squaredLength(const Eigen::MatrixBase<DerivedV> &v) {
	return v.SquaredNorm();
}

template <typename DerivedV>
KRR_DEVICE_FUNCTION auto any(const Eigen::MatrixBase<DerivedV> &v) {
	return v.any();
}

// overload binary operators

template <typename DerivedA, typename DerivedB> 
KRR_DEVICE_FUNCTION auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.cross(b);
}

template <typename DerivedA, typename DerivedB>
KRR_DEVICE_FUNCTION auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.dot(b);
}

template <typename DerivedA, typename DerivedB>
KRR_DEVICE_FUNCTION auto operator/(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.cwiseQuotient(b);
}

template <typename T>
KRR_CALLABLE T mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T) ((result < 0) ? result + b : result);
}

template <typename T>
KRR_CALLABLE T safe_sqrt(T value) {
	return sqrt(max((T) 0, value));
}

} // namespace math
KRR_NAMESPACE_END