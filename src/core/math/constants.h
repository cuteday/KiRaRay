#pragma once

#include <limits>
#ifdef __CUDACC__
#include <math_constants.h>
#endif
#include "common.h"

#ifndef M_PI
#define M_PI			3.14159265358979323846f
#endif
#define M_2PI			6.28318530717958647693f
#define M_4PI			12.5663706143591729539f 

#define M_INV_PI		0.318309886183790671538f    
#define M_INV_2PI		0.15915494309189533577f    
#define M_INV_4PI		0.07957747154594766788f

#define M_EPSILON		1e-5f


#ifdef min
#undef max
#undef min
#endif