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

# define KRR_INTERFACE /* nothing - currently not building any special 'owl.dll' */

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

#if defined(__CUDA_ARCH__)
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
  namespace common {

#ifdef __CUDA_ARCH__
    using ::min;
    using ::max;
    // inline __both__ float abs(float f)      { return fabsf(f); }
    // inline __both__ double abs(double f)    { return fabs(f); }
    using std::abs;
    // inline __both__ float sin(float f) { return ::sinf(f); }
    // inline __both__ double sin(double f) { return ::sin(f); }
    // inline __both__ float cos(float f) { return ::cosf(f); }
    // inline __both__ double cos(double f) { return ::cos(f); }

    using ::saturate;
#else
    using std::min;
    using std::max;
    using std::abs;
    // inline __both__ double sin(double f) { return ::sin(f); }
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
    
  } 
} 