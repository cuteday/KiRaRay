  #pragma once

#include "math/vec/functors.h"
#include "math/vec/compare.h"
#include "math/vec/rotate.h"
#include "math/vec.h"
#include "math/quat.h"
#include "math/aabb.h"
#include "math/mat.h"
#include "math/complex.h"
#include "math/constants.h"
#include "math/transform.h"

KRR_NAMESPACE_BEGIN

using color = vec3f;
using color3f = vec3f;
using point = vec3f;
using point3f = vec3f;
using point2f = vec2f;
using AABB = aabb3f;

template <typename T>
KRR_CALLABLE T mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template <typename T, int n>
KRR_CALLABLE T average(math::vec_t<T, n> v) {
    float val{};
    for (int i = 0; i < n; i++) {
        val += v[i];
    }
    return val / n;
}

template <typename T, int n>
KRR_CALLABLE T isValid(math::vec_t<T, n> v) {
    return !isnan(v) && !isinf(v);
}

KRR_NAMESPACE_END