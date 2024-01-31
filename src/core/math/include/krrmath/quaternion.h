#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "vector.h"

NAMESPACE_BEGIN(krr)

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

	std::string string() const {
		std::stringstream ss;
		ss << *this;
		return ss.str();
	}

#ifdef KRR_MATH_JSON
	friend void to_json(json &j, const Quaternion<T> &q) { 
		j.push_back(q.w());
		j.push_back(q.x());
		j.push_back(q.y());
		j.push_back(q.z());
	}

	friend void from_json(const json &j, Quaternion<T> &q) {
		assert(j.size() == 4);
		q.w() = (T) j.at(0);
		q.x() = (T) j.at(1);
		q.y() = (T) j.at(2);
		q.z() = (T) j.at(3);
	}
#endif
};

using Quaternionf = Quaternion<float>;

NAMESPACE_END(krr)