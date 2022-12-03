#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "vector.h"

KRR_NAMESPACE_BEGIN

template <typename T, int Rows, int Cols, int Options = Eigen::RowMajor>
class Matrix : public Eigen::Matrix<T, Rows, Cols, Options> {
public:
	using Eigen::Matrix<T, Rows, Cols, Options>::Matrix;
	KRR_CALLABLE Matrix(void)
		: Eigen::Matrix<T, Rows, Cols, Options>(Eigen::Matrix<T, Rows, Cols, Options>::Zero()) {}

	template <typename OtherDerived>
	KRR_CALLABLE Matrix(const Eigen::MatrixBase<OtherDerived> &other)
		: Eigen::Matrix<T, Rows, Cols, Options>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Matrix &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
		this->Eigen::Matrix<T, Rows, Cols, Options>::operator=(other);
		return *this;
	}
};

template <typename T, int Options = Eigen::RowMajor>
class Transform : public Matrix<T, 4, 4> {
public:
	using Matrix<T, 4, 4>::Matrix;

	KRR_CALLABLE Matrix<T, 3, 3> rotation() const {
		return Matrix<T, 3, 3> { 
			{ this->operator()(0, 0), this->operator()(0, 1), this->operator()(0, 2) },
			{ this->operator()(1, 0), this->operator()(1, 1), this->operator()(1, 2) },
			{ this->operator()(2, 0), this->operator()(2, 1), this->operator()(2, 2) }
		};
	}

	KRR_CALLABLE Vector3<T> translation() const {
		return { this->operator()(0, 3), this->operator()(1, 3), this->operator()(2, 3) };
	}
};

template <int Rows, int Cols, int Options = Eigen::RowMajor>
using Matrixf = Matrix<float, Rows, Cols, Options>;

template <int Options = Eigen::RowMajor>
using Transformf = Transform<float, Options>;

KRR_NAMESPACE_END