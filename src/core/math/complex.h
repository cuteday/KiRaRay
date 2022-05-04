#pragma once

#include "common.h"
#include "constants.h"
#include <iostream>
#include <math.h>
#include <algorithm>

KRR_NAMESPACE_BEGIN
namespace math
{
	template <typename T>
	struct complex {
		KRR_CALLABLE complex(T re) : re(re), im(0) {}
		KRR_CALLABLE complex(T re, T im) : re(re), im(im) {}

		KRR_CALLABLE complex operator-() const { return { -re, -im }; }

		KRR_CALLABLE complex operator+(complex z) const { return { re + z.re, im + z.im }; }

		KRR_CALLABLE complex operator-(complex z) const { return { re - z.re, im - z.im }; }

		KRR_CALLABLE complex operator*(complex z) const {
			return { re * z.re - im * z.im, re * z.im + im * z.re };
		}

		KRR_CALLABLE complex operator/(complex z) const {
			T scale = 1 / (z.re * z.re + z.im * z.im);
			return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
		}

		friend KRR_CALLABLE complex operator+(T value, complex z) {
			return complex(value) + z;
		}

		friend KRR_CALLABLE complex operator-(T value, complex z) {
			return complex(value) - z;
		}

		friend KRR_CALLABLE complex operator*(T value, complex z) {
			return complex(value) * z;
		}

		friend KRR_CALLABLE complex operator/(T value, complex z) {
			return complex(value) / z;
		}

		KRR_CALLABLE T norm() { return re * re + im * im; }
		T re, im;
	};

	template <typename T>
	KRR_CALLABLE T real(const complex<T>& z) {
		return z.re;
	}

	template <typename T>
	KRR_CALLABLE T imag(const complex<T>& z) {
		return z.im;
	}

	template <typename T>
	KRR_CALLABLE T norm(const complex<T>& z) {
		return z.re * z.re + z.im * z.im;
	}

	template <typename T>
	KRR_CALLABLE T abs(const complex<T>& z) {
		return sqrt(norm(z));
	}

	template <typename T>
	KRR_CALLABLE complex<T> sqrt(const complex<T>& z) {
		T n = abs(z), t1 = sqrt(T(.5) * (n + abs(z.re))),
			t2 = T(.5) * z.im / t1;

		if (n == 0)
			return 0;

		if (z.re >= 0)
			return { t1, t2 };
		else
			return { abs(t2), copysign(t1, z.im) };
	}
}

KRR_NAMESPACE_END