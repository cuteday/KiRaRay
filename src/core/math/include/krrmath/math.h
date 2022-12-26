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
#include "complex.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

KRR_NAMESPACE_BEGIN

using Color = Array3f;
using Color3f = Array3f;
using Color4f = Array4f;
using Point = Vector3f;
using Point2f = Vector2f;
using Point3f = Vector3f;
using AABB = AABB3f;

KRR_NAMESPACE_END