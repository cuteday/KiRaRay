#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#include "logger.h"
#include "device/cuda.h"

#define OPTIX_CHECK( call )													\
	do {																	\
		OptixResult res = call;                                             \
		if( res != OPTIX_SUCCESS )                                          \
		{																	\
			fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
			throw std::runtime_error("OptiX check failed");                 \
		}																	\
	} while (false)

#define OPTIX_CHECK_WITH_LOG(EXPR, LOG)                                             \
	do {                                                                            \
		OptixResult res = EXPR;                                                     \
		if (res != OPTIX_SUCCESS)                                                   \
		{                                                                           \
			fprintf(stderr, "OptiX call " #EXPR " failed with code %d: \"%s\"\nLogs: %s", \
				int(res), optixGetErrorString(res), LOG);                           \
				throw std::runtime_error("OptiX check failed");                     \
		}                                                                           \
	} while (false) /* eat semicolon */

