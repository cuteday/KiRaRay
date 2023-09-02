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

	KRR_CALLABLE bool intersect(Vector3f o, Vector3f d, 
		float tMax = std::numeric_limits<float>::max(), 
		float* tHit0 = nullptr, float* tHit1 = nullptr) const {
		float t0 = 0, t1 = tMax;
		for (int i = 0; i < 3; ++i) {
			// Update interval for _i_th bounding box slab
			float invRayDir = 1 / d[i];
			float tNear		= (min()[i] - o[i]) * invRayDir;
			float tFar		= (max()[i] - o[i]) * invRayDir;
			// Update parametric interval from slab intersection $t$ values
			if (tNear > tFar) std::swap(tNear, tFar);
			// Update _tFar_ to ensure robust ray--bounds intersection
			tFar *= 1 + 2 * gamma(3);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) return false;
		}
		if (tHit0) *tHit0 = t0;
		if (tHit1) *tHit1 = t1;
		return true;
	}
};
	
using AABB3f = AxisAligned<float, 3>;

KRR_NAMESPACE_END