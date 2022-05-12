#pragma once
#include "common.h"

KRR_NAMESPACE_BEGIN

static constexpr float OneMinusEpsilon = 0x1.fffffep-1;
static constexpr int PrimeTableSize = 1000;
extern KRR_DEVICE const int Primes[PrimeTableSize];


KRR_NAMESPACE_END