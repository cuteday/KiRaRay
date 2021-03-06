#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "common.h"
#include "math/vector.h"

KRR_NAMESPACE_BEGIN
	
template <typename T, int Size>
class AxisAligned : public Eigen::AlignedBox<T, Size> {
public:
	using Eigen::AlignedBox<T, Size>::AlignedBox;
	
	KRR_CALLABLE AxisAligned() : Eigen::AlignedBox<T, Size>() {}

	KRR_CALLABLE AxisAligned(const Eigen::AlignedBox<T, Size> &other) : Eigen::AlignedBox<T, Size>(other) {}

	KRR_CALLABLE AxisAligned &operator=(const Eigen::AlignedBox<T, Size> &other) {
		this->Eigen::AlignedBox<T, Size>::operator=(other);
		return *this;
	}
};
	
using AABB3f = AxisAligned<float, 3>;

KRR_NAMESPACE_END