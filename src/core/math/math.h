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

template <typename T>
__both__ inline T real(const complex<T>& z) {
    return z.re;
}

template <typename T>
__both__ inline T imag(const complex<T>& z) {
    return z.im;
}

template <typename T>
__both__ inline T norm(const complex<T>& z) {
    return z.re * z.re + z.im * z.im;
}

template <typename T>
__both__ inline T abs(const complex<T>& z) {
    return sqrt(norm(z));
}

template <typename T>
__both__ inline complex<T> sqrt(const complex<T>& z) {
    T n = abs(z), t1 = sqrt(T(.5) * (n + abs(z.re))),
        t2 = T(.5) * z.im / t1;

    if (n == 0)
        return 0;

    if (z.re >= 0)
        return { t1, t2 };
    else
        return { abs(t2), copysign(t1, z.im) };
}

KRR_NAMESPACE_END