#pragma once
#include <Eigen/Dense>

#include "common.h"
#include "vector.h"

KRR_NAMESPACE_BEGIN

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


KRR_NAMESPACE_END