#pragma once

#include "common.h"
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

template <typename T, int Size>
class Array : public Eigen::Array<T, Size, 1> {
public:
	using Eigen::Array<T, Size, 1>::Array;

	KRR_CALLABLE Array(void) : Eigen::Array<T, Size, 1>() {}

	template <typename OtherDerived>
	KRR_CALLABLE Array(const Eigen::DenseBase<OtherDerived> &other) : Eigen::Array<T, Size, 1>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Array &operator=(const Eigen::DenseBase<OtherDerived> &other) {
		this->Eigen::Array<T, Size, 1>::operator=(other);
		return *this;
	}

	template <typename OtherDerived>
	KRR_CALLABLE Array(const Eigen::Array<OtherDerived, Size, 1> &other) {
		*this = other.template cast<T>();
	}
};

using arr3f = Array<float, 3>;

KRR_NAMESPACE_END