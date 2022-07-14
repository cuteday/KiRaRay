#pragma once

#include <type_traits>
#include "math/vec.h"

namespace krr {
namespace math {

// power shortcuts
template <typename T> __both__ constexpr T pow1(T x) { return x; }
template <typename T> __both__ constexpr T pow2(T x) { return x * x; }
template <typename T> __both__ constexpr T pow3(T x) { return x * x * x; }
template <typename T> __both__ constexpr T pow4(T x) { return x * x * x * x; }
template <typename T> __both__ constexpr T pow5(T x) { return x * x * x * x * x; }

template <typename T> inline __both__ T clamp(const T &val, const T &lo, const T &hi) { return min(hi, max(lo, val)); }

template <typename T> inline __both__ T clamp(const T &val, const T &hi) { return clamp(val, (T) 0, hi); }


/*! helper function that creates a semi-random color from an ID */
inline __both__ vec3f randomColor(int i) {
	int r = unsigned(i) * 13 * 17 + 0x234235;
	int g = unsigned(i) * 7 * 3 * 5 + 0x773477;
	int b = unsigned(i) * 11 * 19 + 0x223766;
	return vec3f((r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f);
}

/*! helper function that creates a semi-random color from an ID */
inline __both__ vec3f randomColor(size_t idx) {
	unsigned int r = (unsigned int) (idx * 13 * 17 + 0x234235);
	unsigned int g = (unsigned int) (idx * 7 * 3 * 5 + 0x773477);
	unsigned int b = (unsigned int) (idx * 11 * 19 + 0x223766);
	return vec3f((r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f);
}

/*! helper function that creates a semi-random color from an ID */
template <typename T> inline __both__ vec3f randomColor(const T *ptr) { return randomColor((size_t) ptr); }


inline __both__ float sqrt(const float v) { return sqrtf(v); }
inline __both__ vec2f sqrt(const vec2f v) { return vec2f(sqrtf(v[0]), sqrtf(v[1])); }
inline __both__ vec3f sqrt(const vec3f v) { return vec3f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2])); }
inline __both__ vec4f sqrt(const vec4f v) { return vec4f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v.w)); }

} // namespace math
} // namespace krr
