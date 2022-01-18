#pragma once

#include "math/math.h"

namespace krr {
  using namespace math;

  struct LaunchParams
  {
    int       frameID { 0 };
    uint32_t *colorBuffer;
    vec2i     fbSize;
  };

}
