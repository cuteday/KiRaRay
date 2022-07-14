#pragma once

namespace krr {
  namespace math {

    /*! perform 'rotation' of float a by amount b. Both a and b must be
      in 0,1 range, the result will be (a+1) clamped to that same
      range (ie, it is the value a shifted by the amount b to the
      right, and re-entering the [0,1) range on the left if it
      "rotates" out on the right */
    inline __both__ float rotate(const float a, const float b)
    {
      float sum = a+b;
      return ((sum-1.f)<0.f)?(sum):(sum-1.f);
    }

    /*! perform 'rotation' of float a by amount b. Both a and b must be
      in 0,1 range, the result will be (a+1) clamped to that same
      range (ie, it is the value a shifted by the amount b to the
      right, and re-entering the [0,1) range on the left if it
      "rotates" out on the right */
    inline __both__ vec2f rotate(const vec2f a, const vec2f b) 
    { return vec2f(rotate(a.x,b.x),rotate(a.y,b.y)); }
  
  } 
} 