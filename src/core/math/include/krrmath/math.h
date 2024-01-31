#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "common.h"
#include "constants.h"
#include "vector.h"
#include "array.h"
#include "matrix.h"
#include "quaternion.h"
#include "aabb.h"
#include "functors.h"
#include "clipspace.h"
#include "transform.h"
#include "complex.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

NAMESPACE_BEGIN(krr)

using Point2f = Vector2f;
using Point3f = Vector3f;
using AABB = AABB3f;

NAMESPACE_END(krr)