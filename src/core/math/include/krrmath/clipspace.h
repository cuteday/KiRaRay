// matrix transformation and clipspace arithmetric
#pragma once
#include <Eigen/Dense>

#include "common.h"
#include "vector.h"
#include "matrix.h"
#include "constants.h"

// Adapted from GLM's clip space transform implementation.

NAMESPACE_BEGIN(krr)

#if KRR_CLIPSPACE_RIGHTHANDED

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> perspective(T fovy, T aspect, T zNear, T zFar) {
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

	T const tanHalfFovy = tan(fovy / static_cast<T>(2));
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Zero()};

	result(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovy);
	result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
	result(3, 2) = -static_cast<T>(1);
#if KRR_CLIPSPACE_Z_FROM_ZERO
	result(2, 2) = -zFar / (zFar - zNear);
	result(2, 3) = -(zFar * zNear) / (zFar - zNear);
#else 
	result(2, 2) = -(zFar + zNear) / (zFar - zNear);
	result(2, 3) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
#endif
	return result;
}

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> orthogonal(T left, T right, T bottom, T top, T zNear, T zFar) {
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};

	result(0, 0) = static_cast<T>(2) / (right - left);
	result(1, 1) = static_cast<T>(2) / (top - bottom);
	result(0, 3) = -(right + left) / (right - left);
	result(1, 3) = -(top + bottom) / (top - bottom);
#if KRR_CLIPSPACE_Z_FROM_ZERO
	result(2, 2) = -static_cast<T>(1) / (zFar - zNear);
	result(2, 3) = -zNear / (zFar - zNear);
#else
	result(2, 2) = -static_cast<T>(2) / (zFar - zNear);
	result(2, 3) = -(zFar + zNear) / (zFar - zNear);
#endif
	return result;
}

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> look_at(Vector3<T> const &eye, Vector3<T> const &center,
								 Vector3<T> const &up) {
	Vector3<T> const f(normalize(center - eye));
	Vector3<T> const s(normalize(cross(f, up)));
	Vector3<T> const u(cross(s, f));

	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};
	result(0, 0) = s[0];
	result(0, 1) = s[1];
	result(0, 2) = s[2];
	result(1, 0) = u[0];
	result(1, 1) = u[1];
	result(1, 2) = u[2];
	result(2, 0) = -f[0];
	result(2, 1) = -f[1];
	result(2, 2) = -f[2];
	result(0, 3) = -dot(s, eye);
	result(1, 3) = -dot(u, eye);
	result(2, 3) = dot(f, eye);
	return result;
}

#else

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> perspective(T fovy, T aspect, T zNear, T zFar) {
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

	T const tanHalfFovy = tan(fovy / static_cast<T>(2));
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Zero()};

	result(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovy);
	result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
	result(3, 2) = static_cast<T>(1);
#if KRR_CLIPSPACE_Z_FROM_ZERO
	result(2, 2) = zFar / (zFar - zNear);
	result(2, 3) = -(zFar * zNear) / (zFar - zNear);
#else
	result(2, 2) = (zFar + zNear) / (zFar - zNear);
	result(2, 3) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
#endif
	return result;
}

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> orthogonal(T left, T right, T bottom, T top, T zNear,
												 T zFar) {
	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};

	result(0, 0) = static_cast<T>(2) / (right - left);
	result(1, 1) = static_cast<T>(2) / (top - bottom);
	result(0, 3) = -(right + left) / (right - left);
	result(1, 3) = -(top + bottom) / (top - bottom);
#if KRR_CLIPSPACE_Z_FROM_ZERO
	result(2, 2) = static_cast<T>(1) / (zFar - zNear);
	result(2, 3) = -zNear / (zFar - zNear);
#else
	result(2, 2) = static_cast<T>(2) / (zFar - zNear);
	result(2, 3) = -(zFar + zNear) / (zFar - zNear);
#endif
	return result;
}

template <typename T, int Options = math::ColMajor>
KRR_CALLABLE Matrix<T, 4, 4, Options> look_at(Vector3<T> const &eye, Vector3<T> const &center,
								 Vector3<T> const &up) {
	Vector3<T> const f(normalize(center - eye));
	Vector3<T> const s(normalize(cross(up, f)));
	Vector3<T> const u(cross(f, s));

	Matrix<T, 4, 4, Options> result{Matrix<T, 4, 4, Options>::Identity()};
	result(0, 0) = s[0];
	result(0, 1) = s[1];
	result(0, 2) = s[2];
	result(1, 0) = u[0];
	result(1, 1) = u[1];
	result(1, 2) = u[2];
	result(2, 0) = f[0];
	result(2, 1) = f[1];
	result(2, 2) = f[2];
	result(0, 3) = -dot(s, eye);
	result(1, 3) = -dot(u, eye);
	result(2, 3) = -dot(f, eye);
	return result;
}

#endif

NAMESPACE_END(krr)