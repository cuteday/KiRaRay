#pragma once

//#include "math/vec/functors.h"
//#include "math/vec/compare.h"
//#include "math/vec/rotate.h"
//#include "math/vec.h"
//#include "math/quat.h"
//#include "math/aabb.h"
//#include "math/mat.h"
//#include "math/complex.h"
#include "math/constants.h"
//#include "math/transform.h"
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

#define normalize(a) (a.normalized())
#define cross(a, b) (a.cross(b))
#define dot(a, b) (a.dot(b))

using vec2f = Eigen::Vector2f;
using vec2i = Eigen::Vector2i;
using vec2ui = Eigen::Vector<uint, 2>;
using vec3f = Eigen::Vector3f;
using vec3i = Eigen::Vector3i;
using vec3ui = Eigen::Vector<uint, 3>;
using vec4f = Eigen::Vector4f;
using vec4i = Eigen::Vector4i;
using vec4ui = Eigen::Vector<uint, 4>;

using color = vec3f;
using color3f = vec3f;
using point = vec3f;
using point3f = vec3f;
using point2f = vec2f;
//using AABB = aabb3f;

template <typename T>
KRR_CALLABLE T mod(T a, T b) {
	T result = a - (a / b) * b;
	vec3f c;	
	return (T)((result < 0) ? result + b : result);
}

template <typename T> KRR_CALLABLE T safe_sqrt(T value) {
	return sqrt(max((T)0, value));
}

KRR_NAMESPACE_END