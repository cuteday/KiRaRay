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

#define NAMESPACE_KRR_BEGIN namespace krr {
#define NAMESPACE_KRR_END }	

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

#ifdef WIN32
# define KRR_TERMINAL_RED ""
# define KRR_TERMINAL_GREEN ""
# define KRR_TERMINAL_LIGHT_GREEN ""
# define KRR_TERMINAL_YELLOW ""
# define KRR_TERMINAL_BLUE ""
# define KRR_TERMINAL_LIGHT_BLUE ""
# define KRR_TERMINAL_RESET ""
# define KRR_TERMINAL_DEFAULT KRR_TERMINAL_RESET
# define KRR_TERMINAL_BOLD ""

# define KRR_TERMINAL_MAGENTA ""
# define KRR_TERMINAL_LIGHT_MAGENTA ""
# define KRR_TERMINAL_CYAN ""
# define KRR_TERMINAL_LIGHT_RED ""
#else
# define KRR_TERMINAL_RED "\033[0;31m"
# define KRR_TERMINAL_GREEN "\033[0;32m"
# define KRR_TERMINAL_LIGHT_GREEN "\033[1;32m"
# define KRR_TERMINAL_YELLOW "\033[1;33m"
# define KRR_TERMINAL_BLUE "\033[0;34m"
# define KRR_TERMINAL_LIGHT_BLUE "\033[1;34m"
# define KRR_TERMINAL_RESET "\033[0m"
# define KRR_TERMINAL_DEFAULT KRR_TERMINAL_RESET
# define KRR_TERMINAL_BOLD "\033[1;1m"

# define KRR_TERMINAL_MAGENTA "\e[35m"
# define KRR_TERMINAL_LIGHT_MAGENTA "\e[95m"
# define KRR_TERMINAL_CYAN "\e[36m"
# define KRR_TERMINAL_LIGHT_RED "\033[1;31m"
#endif

#ifdef _MSC_VER
# define KRR_ALIGN(alignment) __declspec(align(alignment)) 
#else
# define KRR_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif

namespace owl {
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

    // inline __both__ float abs(float f)      { return fabsf(f); }
    // inline __both__ double abs(double f)    { return fabs(f); }
    inline __both__ float rcp(float f)      { return 1.f/f; }
    inline __both__ double rcp(double d)    { return 1./d; }
  
    inline __both__ int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
    inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
    inline __both__ int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
    inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }
  
    using ::sin; // this is the double version
    using ::cos; // this is the double version
    
#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  
    /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
    inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of thousands (K), millions
        (M), etc. Ie, the value 64000 would be returned as 64K, and
        65536 would be 65.5K */
    inline std::string prettyNumber(const size_t s)
    {
      char buf[1000];
      if (s >= (1000LL*1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
      } else if (s >= (1000LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
        "3400000", etc, using mulitples of 1024 as in kilobytes,
        etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
    inline std::string prettyBytes(const size_t s)
    {
      char buf[1000];
      if (s >= (1024LL*1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
      } else if (s >= (1024LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    inline bool hasSuffix(const std::string &s, const std::string &suffix)
    {
      return s.substr(s.size()-suffix.size()) == suffix;
    }
    
  } 
} 