#include "common.h"
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

KRR_NAMESPACE_END