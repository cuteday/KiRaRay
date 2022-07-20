#pragma once

#include "common.h"
#include "math/vector.h"

KRR_NAMESPACE_BEGIN

namespace math {

#ifdef KRR_DEVICE_CODE
using ::min;
using ::max;
using ::abs;
using ::fmod;
using ::copysign;
#else
using std::min;
using std::max;
using std::abs;
using std::fmod;
using std::copysign;	
#endif
	
using ::sin;
using ::cos;
using ::pow;

// power shortcuts
template <typename T>
KRR_CALLABLE constexpr T pow1(T x) {
	return x;
}
template <typename T>
KRR_CALLABLE constexpr T pow2(T x) {
	return x * x;
}
template <typename T>
KRR_CALLABLE constexpr T pow3(T x) {
	return x * x * x;
}
template <typename T>
KRR_CALLABLE constexpr T pow4(T x) {
	return x * x * x * x;
}
template <typename T>
KRR_CALLABLE constexpr T pow5(T x) {
	return x * x * x * x * x;
}

KRR_CALLABLE float sqrt(const float v) { return sqrtf(v); }
KRR_CALLABLE Vec2f sqrt(const Vec2f v) { return Vec2f(sqrtf(v[0]), sqrtf(v[1])); }
KRR_CALLABLE Vec3f sqrt(const Vec3f v) { return Vec3f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2])); }
KRR_CALLABLE Vec4f sqrt(const Vec4f v) { return Vec4f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v[3])); }

template <typename T>
KRR_CALLABLE T mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T) ((result < 0) ? result + b : result);
}

template <typename T>
KRR_CALLABLE T safe_sqrt(T value) {
	return sqrt(max((T) 0, value));
}

KRR_CALLABLE float saturate(const float &f) { return min(1.f, max(0.f, f)); }

KRR_CALLABLE float rcp(float f) { return 1.f / f; }

} // namespace math

KRR_NAMESPACE_END