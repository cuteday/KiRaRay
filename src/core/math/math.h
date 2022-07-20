#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "common.h"
#include "math/constants.h"
#include "math/vector.h"
#include "math/array.h"
#include "math/quaternion.h"
#include "math/aabb.h"
#include "math/functors.h"
#include "math/complex.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

KRR_NAMESPACE_BEGIN

using Color = Array3f;
using Color3f = Array3f;
using Color4f = Array4f;
using Point = Vec3f;
using Point2f = Vec2f;
using Point3f = Vec3f;
using AABB = AABB3f;
using Quat = Quaternionf;

namespace math {

template <typename T>
KRR_CALLABLE auto clamp(T v, T lo, T hi) {
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
KRR_CALLABLE auto lerp(const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b, DerivedT t) {
	return (a.eval() * (1 - t) + b.eval() * t).eval();
}

// overload unary opeartors

template <typename DerivedV>
KRR_CALLABLE auto normalize(const Eigen::MatrixBase<DerivedV> &v) {
	return v.normalized();
}

template <typename DerivedV>
KRR_CALLABLE auto abs(const Eigen::MatrixBase<DerivedV> &v) {
	return v.cwiseAbs();
}

template <typename DerivedV>
KRR_CALLABLE auto length(const Eigen::MatrixBase<DerivedV> &v) {
	return v.norm();
}

template <typename DerivedV>
KRR_CALLABLE auto squaredLength(const Eigen::MatrixBase<DerivedV> &v) {
	return v.SquaredNorm();
}

template <typename DerivedV>
KRR_CALLABLE auto any(const Eigen::DenseBase<DerivedV> &v) {
	return v.any();
}

// overload binary operators

template <typename DerivedA, typename DerivedB> 
KRR_CALLABLE auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.cross(b);
}

template <typename DerivedA, typename DerivedB>
KRR_CALLABLE auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.dot(b);
}

template <typename DerivedA, typename DerivedB>
KRR_CALLABLE auto operator/(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
	return a.cwiseQuotient(b);
}

} // namespace math

KRR_NAMESPACE_END