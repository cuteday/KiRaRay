#pragma once

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

template <typename DerivedV, typename DerivedB>
auto clamp(const DerivedV &v, DerivedB lo, DerivedB hi) {
	return max(min(v, hi), lo);
}

template <typename DerivedV, typename DerivedB>
auto clamp(const Eigen::MatrixBase<DerivedV> &v, DerivedB lo, DerivedB hi) {
	return v.cwiseMin(hi).cwiseMax(lo);
	Aabb3f aabb;
}

template <typename DerivedV, typename DerivedB>
auto lerp(const Eigen::MatrixBase<DerivedV> &v, DerivedB t) {
	return v * (1 - t) + t;
}

// overload unary opeartors

template <typename DerivedV>
auto normalize(const Eigen::MatrixBase<DerivedV> &v) {
	return v.normalized();
}

template <typename DerivedV>
auto abs(const Eigen::MatrixBase<DerivedV> &v) {
	return v.cwiseAbs();
}

template <typename DerivedV>
auto length(const Eigen::MatrixBase<DerivedV> &v) {
	return v.norm();
}

template <typename DerivedV>
auto squaredLength(const Eigen::MatrixBase<DerivedV> &v) {
	return v.SquaredNorm();
}

template <typename DerivedV>
auto any(const Eigen::MatrixBase<DerivedV> &v) {
	return v.any();
}

// overload binary operators

template <typename DerivedV>
auto cross(const Eigen::MatrixBase<DerivedV> &a, const Eigen::MatrixBase<DerivedV> &b) {
	return a.cross(b);
}

template <typename DerivedV>
auto dot(const Eigen::MatrixBase<DerivedV> &a, const Eigen::MatrixBase<DerivedV> &b) {
	return a.dot(b);
}

template <typename DerivedV>
auto operator / (const Eigen::MatrixBase<DerivedV> &a, const Eigen::MatrixBase<DerivedV> &b) {
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