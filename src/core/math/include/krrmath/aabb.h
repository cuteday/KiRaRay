#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "common.h"
#include "vector.h"

KRR_NAMESPACE_BEGIN
	
template <typename T, int Size>
class AxisAligned : public Eigen::AlignedBox<T, Size> {
public:
	using VectorType = Vector<T, Size>;
	using Eigen::AlignedBox<T, Size>::AlignedBox;
	
	KRR_CALLABLE AxisAligned() : Eigen::AlignedBox<T, Size>() {}

	KRR_CALLABLE AxisAligned(T min, T max) :
		Eigen::AlignedBox<T, Size>(VectorType::Constant(min), VectorType::Constant(max)) {}

	KRR_CALLABLE AxisAligned(const Eigen::AlignedBox<T, Size> &other) : Eigen::AlignedBox<T, Size>(other) {}

	KRR_CALLABLE AxisAligned &operator=(const Eigen::AlignedBox<T, Size> &other) {
		this->Eigen::AlignedBox<T, Size>::operator=(other);
		return *this;
	}

	template <int Mode, int Options>
	KRR_CALLABLE AxisAligned operator*(const Eigen::Transform<T, Size, Mode, Options> & t) 
	{ return this->transformed(t); }

	KRR_CALLABLE VectorType min() const { return Eigen::AlignedBox<T, Size>::min(); }
	KRR_CALLABLE VectorType max() const { return Eigen::AlignedBox<T, Size>::max(); }

	KRR_CALLABLE void inflate(T inflation) {
		this->m_min -= VectorType::Constant(inflation);
		this->m_max += VectorType::Constant(inflation);
	}

	KRR_CALLABLE VectorType clip(const VectorType& p) const {
		return p.cwiseMin(this->m_max).cwiseMax(this->m_min);
	}
};
	
using AABB3f = AxisAligned<float, 3>;

KRR_NAMESPACE_END