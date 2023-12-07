#pragma once

#include "common.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN

#define CUDA_CHECK(call)                                                                           \
	do {                                                                                           \
		cudaError_t rc = call;                                                                     \
		if (rc != cudaSuccess) {                                                                   \
			cudaError_t err = rc; /*cudaGetLastError();*/                                          \
			Log(Fatal, "CUDA Error (%s: line %d): %s (%s)\n", __FILE__, __LINE__,                  \
				cudaGetErrorName(err), cudaGetErrorString(err));                                   \
		}                                                                                          \
	} while (0)

#define CUDA_SYNC(call)                                                                            \
	do {                                                                                           \
		cudaDeviceSynchronize();                                                                   \
		call;                                                                                      \
		cudaDeviceSynchronize();                                                                   \
	} while (0)

#define CUDA_SYNC_CHECK()                                                                          \
	do {                                                                                           \
		cudaDeviceSynchronize();                                                                   \
		cudaError_t error = cudaGetLastError();                                                    \
		if (error != cudaSuccess) {                                                                \
			Log(Fatal, "Error (%s: line %d): %s\n", __FILE__, __LINE__,                            \
				cudaGetErrorString(error));                                                        \
		}                                                                                          \
	} while (0)

#define OPTIX_CHECK(call)                                                                          \
	do {                                                                                           \
		OptixResult res = call;                                                                    \
		if (res != OPTIX_SUCCESS) {                                                                \
			Log(Fatal, "Optix call (%s) failed with code %d (line %d)\n", #call, res,              \
					__LINE__);                                                                     \
		}                                                                                          \
	} while (false)

#define OPTIX_CHECK_WITH_LOG(EXPR, LOG)                                                            \
	do {                                                                                           \
		OptixResult res = EXPR;                                                                    \
		if (res != OPTIX_SUCCESS) {                                                                \
			Log(Fatal, "Optix call " #EXPR " failed with code %d: \"%s\"\nLogs: %s",               \
					int(res), optixGetErrorString(res), LOG);                                      \
		}                                                                                          \
	} while (false) /* eat semicolon */


#define CHECK_LOG(EXPR, LOG, ...)                                                                  \
	do {                                                                                           \
		if (!(EXPR)) {                                                                             \
			Log(Fatal, "Error (%s: line %d): "##LOG, __FILE__, __LINE__, ##__VA_ARGS__);           \
		}                                                                                          \
	} while (0)

	
#define CHECK(x) assert(x)
#define CHECK_IMPL(a, b, op) assert((a)op(b))

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)


#ifdef KRR_DEBUG_BUILD

#define DCHECK(x) (CHECK(x))
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)

#else

#define EMPTY_CHECK \
    do {            \
    } while (false) /* swallow semicolon */

#define DCHECK(x) EMPTY_CHECK

#define DCHECK_EQ(a, b) EMPTY_CHECK
#define DCHECK_NE(a, b) EMPTY_CHECK
#define DCHECK_GT(a, b) EMPTY_CHECK
#define DCHECK_GE(a, b) EMPTY_CHECK
#define DCHECK_LT(a, b) EMPTY_CHECK
#define DCHECK_LE(a, b) EMPTY_CHECK

#endif


KRR_NAMESPACE_END