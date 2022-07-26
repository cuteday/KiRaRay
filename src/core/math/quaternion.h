#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "vector.h"

KRR_NAMESPACE_BEGIN

template <typename T>
class Quaternion : public Eigen::Quaternion<T> {
public:
	using Eigen::Quaternion<T>::Quaternion;

	KRR_CALLABLE Quaternion(void) : Eigen::Quaternion<T>() {}

	template <typename OtherDerived>
	KRR_CALLABLE Quaternion(const Eigen::QuaternionBase<OtherDerived> &other) : Eigen::Quaternion<T>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Quaternion &operator=(const Eigen::QuaternionBase<OtherDerived> &other) {
		this->Eigen::Quaternion<T>::operator=(other);
		return *this;
	}

	KRR_CALLABLE static Quaternion fromAxisAngle(const Vector3f &axis, T angle) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
	}

	KRR_CALLABLE static Quaternion fromEuler(T yaw, T pitch, T roll) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(yaw, krr::Vector3<T>::UnitY()) *
									Eigen::AngleAxis<T>(roll, krr::Vector3<T>::UnitZ()) *
									Eigen::AngleAxis<T>(pitch, krr::Vector3<T>::UnitX()));
	}
};

using Quaternionf = Quaternion<float>;

KRR_NAMESPACE_END