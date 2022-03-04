  #pragma once

#include "math/vec/functors.h"
#include "math/vec/compare.h"
#include "math/vec/rotate.h"
#include "math/vec.h"
#include "math/quat.h"
#include "math/aabb.h"
#include "math/mat.h"
#include "math/transform.h"

KRR_NAMESPACE_BEGIN

template <typename T>
struct complex {
    __both__ inline complex(T re) : re(re), im(0) {}
    __both__ inline complex(T re, T im) : re(re), im(im) {}

    __both__ inline complex operator-() const { return { -re, -im }; }

    __both__ inline complex operator+(complex z) const { return { re + z.re, im + z.im }; }

    __both__ inline complex operator-(complex z) const { return { re - z.re, im - z.im }; }

    __both__ inline complex operator*(complex z) const {
        return { re * z.re - im * z.im, re * z.im + im * z.re };
    }

    __both__ inline complex operator/(complex z) const {
        T scale = 1 / (z.re * z.re + z.im * z.im);
        return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
    }

    friend __both__ inline complex operator+(T value, complex z) {
        return complex(value) + z;
    }

    friend __both__ inline complex operator-(T value, complex z) {
        return complex(value) - z;
    }

    friend __both__ inline complex operator*(T value, complex z) {
        return complex(value) * z;
    }

    friend __both__ inline complex operator/(T value, complex z) {
        return complex(value) / z;
    }

    __both__ inline T norm() { return re * re + im * im; }
    T re, im;
};

KRR_NAMESPACE_END