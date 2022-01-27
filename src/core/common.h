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

using std::string;

typedef uint32_t uint;

#define KRR_PROJECT_NAME "KiRaRay"
#define KRR_NAMESPACE_BEGIN namespace krr {
#define KRR_NAMESPACE_END }
#define KRR_CLASS_SHARED_PTR(name) using SharedPtr = std::shared_ptr<name>;

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

# define KRR_INTERFACE

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

#if defined(__NVCC__)
# define __krr_device   __device__
# define __krr_host     __host__
#else
# define __krr_device   /* ignore */
# define __krr_host     /* ignore */
#endif

# define __both__   __krr_host __krr_device

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

namespace krr {
  namespace math {

#ifdef __NVCC__
     using ::min;
     using ::max;
     using std::abs;
     using ::saturate;
#else
    using std::min;
    using std::max;
    using std::abs;
    inline __both__ float saturate(const float &f) { return min(1.f,max(0.f,f)); }
#endif

    inline __both__ float rcp(float f)      { return 1.f/f; }
    inline __both__ double rcp(double d)    { return 1./d; }
  
    inline __both__ int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
    inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
    inline __both__ int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
    inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }
  
     using ::sin; // this is the double version
     using ::cos; // this is the double version
    
    namespace polymorphic {

#ifndef __NVCC__
        inline __both__ float sqrt(const float f) { return ::sqrtf(f); }
        inline __both__ double sqrt(const double d) { return ::sqrt(d); }

        inline __both__ float rsqrt(const float f) { return 1.f / polymorphic::sqrt(f); }
        inline __both__ double rsqrt(const double d) { return 1. / polymorphic::sqrt(d); }
#endif

    }

  } 
} 