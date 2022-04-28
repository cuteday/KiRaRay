#pragma once

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>

#include "config.h"

using std::string;
using std::to_string;

typedef uint32_t uint;
typedef unsigned char uchar;

#define KRR_NAMESPACE_BEGIN namespace krr {
#define KRR_NAMESPACE_END }

#if defined(_MSC_VER)
#  define KRR_DLL_EXPORT __declspec(dllexport)
#  define KRR_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define KRR_DLL_EXPORT __attribute__((visibility("default")))
#  define KRR_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define KRR_DLL_EXPORT
#  define KRR_DLL_IMPORT
#endif

#define KRR_INTERFACE
#define KRR_RESTRICT	__restrict

#if defined(__NVCC__)
#define KRR_DEVICE_CODE
#endif
#if defined(_MSC_VER)
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

#ifdef KRR_DEVICE_CODE
# define KRR_DEVICE   __device__
# define KRR_HOST     __host__
# define KRR_FORCEINLINE __forceinline__
#else
# define KRR_DEVICE       /* ignore */
# define KRR_HOST         /* ignore */
# define KRR_FORCEINLINE  /* ignore */
#endif

# define __both__   KRR_HOST KRR_DEVICE
# define KRR_CALLABLE inline KRR_HOST KRR_DEVICE
# define KRR_HOST_DEVICE KRR_HOST KRR_DEVICE
# define KRR_DEVICE_FUNCTION KRR_DEVICE KRR_FORCEINLINE
# define KRR_DEVICE_LAMBDA(...) [ =, *this ] KRR_HOST_DEVICE(__VA_ARGS__) mutable


#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#define KRR_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")
#define KRR_SHOULDNT_GO_HERE __assume(0)

#ifdef _MSC_VER
# define KRR_ALIGN(alignment) __declspec(align(alignment)) 
#else
# define KRR_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif

KRR_NAMESPACE_BEGIN

namespace inter {
	template <typename T>
	class polymorphic_allocator;
}
// this allocator uses gpu memory by default.
using Allocator = inter::polymorphic_allocator<std::byte>;

namespace math {
#ifdef KRR_DEVICE_CODE
	using ::min;
	using ::max;
	using ::abs;
	using ::copysign;
#else
	using std::min;
	using std::max;
	using std::abs;
	using std::copysign;	
#endif

	inline __both__ float saturate(const float& f) { return min(1.f, max(0.f, f)); }
	inline __both__ float rcp(float f) { return 1.f / f; }
	inline __both__ double rcp(double d) { return 1. / d; }

	template <typename T>
	inline __both__ T divRoundUp(T val, T divisor) { return (val + divisor - 1) / divisor; }

	using ::sin; // this is the double version
	using ::cos; // this is the double version
	using ::pow;

	namespace polymorphic {
		inline __both__ float sqrt(const float f) { return sqrtf(f); }
		inline __both__ double sqrt(const double d) { return sqrt(d); }

		inline __both__ float rsqrt(const float f) { return 1.f / ::sqrt(f); }
		inline __both__ double rsqrt(const double d) { return 1. / ::sqrt(d); }
	}
}

KRR_NAMESPACE_END