#pragma once

#include "common.h"
#include "math/vec.h"

KRR_NAMESPACE_BEGIN

namespace math {

// power shortcuts
template <typename T> __both__ constexpr T pow1(T x) { return x; }
template <typename T> __both__ constexpr T pow2(T x) { return x * x; }
template <typename T> __both__ constexpr T pow3(T x) { return x * x * x; }
template <typename T> __both__ constexpr T pow4(T x) { return x * x * x * x; }
template <typename T> __both__ constexpr T pow5(T x) { return x * x * x * x * x; }

inline __both__ float sqrt(const float v) { return sqrtf(v); }
inline __both__ vec2f sqrt(const vec2f v) { return vec2f(sqrtf(v[0]), sqrtf(v[1])); }
inline __both__ vec3f sqrt(const vec3f v) { return vec3f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2])); }
inline __both__ vec4f sqrt(const vec4f v) { return vec4f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v[3])); }



} // namespace math

KRR_NAMESPACE_END