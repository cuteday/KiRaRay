  #pragma once

#include "math/vec/functors.h"
#include "math/vec/compare.h"
#include "math/vec/rotate.h"
#include "math/vec.h"
#include "math/quat.h"
#include "math/aabb.h"
#include "math/mat.h"
#include "math/complex.h"
#include "math/transform.h"

KRR_NAMESPACE_BEGIN

using color = vec3f;

template <typename T>
KRR_CALLABLE T mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

KRR_NAMESPACE_END