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

	KRR_CALLABLE static Quaternion fromAxisAngle(const vec3f &axis, T angle) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
	}

	KRR_CALLABLE static Quaternion fromEuler(T yaw, T pitch, T roll) {
		return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(roll, vec3f::UnitZ()) *
									Eigen::AngleAxis<T>(pitch, vec3f::UnitX()) *
									Eigen::AngleAxis<T>(yaw, vec3f::UnitY()));
	}
};

class Quaternionf : public Quaternion<float> {
public:
	using Quaternion<float>::Quaternion;
};



KRR_NAMESPACE_END