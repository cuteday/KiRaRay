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

template <int Rows, int Cols, int Options = Eigen::RowMajor>
using Matrixf = Matrix<float, Rows, Cols, Options>;

KRR_NAMESPACE_END