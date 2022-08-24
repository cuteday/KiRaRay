#pragma once
#include <Eigen/Dense>

#include "common.h"
#include "math/vector.h"

KRR_NAMESPACE_BEGIN

namespace math {

#ifdef KRR_DEVICE_CODE
using ::abs;
using ::copysign;
using ::fmod;
using ::max;
using ::min;
using ::isnan;
#else
using std::abs;
using std::copysign;
using std::fmod;
using std::max;
using std::min;
using std::isnan;
#endif

using ::cos;
using ::pow;
using ::sin;
using ::tan;
using ::sinh;
using ::cosh;
using ::tanh;

// power shortcuts
template <typename T> KRR_CALLABLE constexpr T pow1(T x) { return x; }
template <typename T> KRR_CALLABLE constexpr T pow2(T x) { return x * x; }
template <typename T> KRR_CALLABLE constexpr T pow3(T x) { return x * x * x; }
template <typename T> KRR_CALLABLE constexpr T pow4(T x) { return x * x * x * x; }
template <typename T> KRR_CALLABLE constexpr T pow5(T x) { return x * x * x * x * x; }

KRR_CALLABLE float sqrt(const float v) { return sqrtf(v); }
KRR_CALLABLE Vector2f sqrt(const Vector2f v) {
	return Vector2f(sqrtf(v[0]), sqrtf(v[1]));
}
KRR_CALLABLE Vector3f sqrt(const Vector3f v) {
	return Vector3f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]));
}
KRR_CALLABLE Vector4f sqrt(const Vector4f v) {
	return Vector4f(sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v[3]));
}

template <typename T> KRR_CALLABLE T mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T) ((result < 0) ? result + b : result);
}

template <typename T> KRR_CALLABLE T safe_sqrt(T value) {
	return sqrt(max((T) 0, value));
}

KRR_CALLABLE float saturate(const float &f) { return min(1.f, max(0.f, f)); }

KRR_CALLABLE float rcp(float f) { return 1.f / f; }

KRR_CALLABLE float logistic(const float x) { return 1 / (1.f + expf(-x)); }

KRR_CALLABLE float csch(const float x) { return 1 / sinh(x); }

KRR_CALLABLE float coth(const float x) { return 1 / tanh(x); }

KRR_CALLABLE float sech(const float x) { return 1 / cosh(x); }

} // namespace math

KRR_NAMESPACE_END