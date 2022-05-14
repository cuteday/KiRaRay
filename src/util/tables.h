#pragma once
#include "common.h"

KRR_NAMESPACE_BEGIN

// TODO: whether we can access __device__ variables on host by just using macros like KRR_DEVICE?
// currently the warning (20091) is suppressed manually (i'm an ocd!)

static constexpr float OneMinusEpsilon = 0x1.fffffep-1;
static constexpr int PrimeTableSize = 1000;
extern KRR_DEVICE const int Primes[PrimeTableSize];

KRR_NAMESPACE_END