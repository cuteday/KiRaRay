#pragma once
#define NOMINMAX
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <json.hpp>

#include "config.h"	

#ifdef __INTELLISENSE__
#pragma diag_suppress 40		// suppress lambda error for visual studio
#endif

using std::string;
using std::to_string;
using nlohmann::json;

typedef uint32_t uint;
typedef unsigned char uchar;

class RGB;
class SampledSpectrum;
#if KRR_RENDER_SPECTRAL
typedef SampledSpectrum Spectrum;
#else
typedef RGB Spectrum;
#endif

#define KRR_NAMESPACE_BEGIN namespace krr {
#define KRR_NAMESPACE_END }

#ifdef KRR_DEBUG_BUILD
#   define KRR_DEBUG_SELECT(A, B) A
#else
#   define KRR_DEBUG_SELECT(A, B) B
#endif

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

#if defined(__CUDA_ARCH__)
#define KRR_DEVICE_CODE
#endif

#if defined(__CUDACC__)
# define KRR_DEVICE   __device__
# define KRR_HOST     __host__
# define KRR_FORCEINLINE __forceinline__
# if defined(KRR_DEVICE_CODE)
# define KRR_CONST	__device__ const 
# else
# define KRR_CONST	const
#endif
# define KRR_GLOBAL	__global__
#else
# define KRR_DEVICE			/* ignore */
# define KRR_HOST			/* ignore */
# define KRR_FORCEINLINE	/* ignore */
# define KRR_CONST	const
# define KRR_GLOBAL			/* ignore */
#endif

# define __both__   KRR_HOST KRR_DEVICE
# define KRR_CALLABLE inline KRR_HOST KRR_DEVICE
# define KRR_HOST_DEVICE KRR_HOST KRR_DEVICE
# define KRR_DEVICE_FUNCTION KRR_DEVICE KRR_FORCEINLINE
# define KRR_DEVICE_LAMBDA(...) [ =, *this ] KRR_DEVICE(__VA_ARGS__) mutable 

#if !defined(__CUDA_ARCH__)
extern const uint3 threadIdx, blockIdx;
extern const dim3 blockDim, gridDim;
#endif	// eliminate intellisense warnings for these kernel built-in variables

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

#define KRR_CLASS_DEFINE NLOHMANN_DEFINE_TYPE_INTRUSIVE
#define KRR_ENUM_DEFINE NLOHMANN_JSON_SERIALIZE_ENUM

#define KRR_ENUM_OPERATORS(name)                                                \
    inline name operator | (name a, name b)                                     \
    { return name(uint32_t(a) | uint32_t(b)); }                                 \
    inline name operator & (name a, name b)                                     \
    { return name(uint32_t(a) & uint32_t(b)); }                                 \
    inline name operator ~ (name a)                                             \
    { return name(~uint32_t(a)); }                                              \
    inline name operator |= (name& a, name b)                                   \
    { a = name(uint32_t(a) | uint32_t(b)); return a; }                          \
    inline name operator &= (name& a, name b)                                   \
    { a = name(uint32_t(a) & uint32_t(b)); return a; }                          \
    inline bool operator !(name a) { return uint32_t(a) == 0; }                 \
    inline bool operator ==(name a, uint32_t b) { return uint32_t(a) == b; }    \
    inline bool operator !=(name a, uint32_t b) { return uint32_t(a) != b; }    

#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

namespace gpu {
	template <typename T>
	class polymorphic_allocator;
}
// this allocator uses gpu memory by default.
using Allocator = gpu::polymorphic_allocator<std::byte>;

KRR_NAMESPACE_END