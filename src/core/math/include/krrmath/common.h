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

#ifndef KRR_NAMESPACE_BEGIN

typedef uint32_t uint;
typedef unsigned char uchar;

#define KRR_NAMESPACE_BEGIN namespace krr {
#define KRR_NAMESPACE_END }

#if defined(__CUDA_ARCH__)
#define KRR_DEVICE_CODE
#endif

#if defined(__CUDACC__)
# define KRR_DEVICE   __device__
# define KRR_HOST     __host__
# define KRR_FORCEINLINE __forceinline__
# define KRR_CONST	__device__ const 
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

#endif

#if defined(INCLUDE_NLOHMANN_JSON_HPP_)
#define KRR_MATH_JSON
#endif
