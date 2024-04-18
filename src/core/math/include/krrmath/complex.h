#pragma once

#include "common.h"
#include "constants.h"
#include <iostream>
#include <math.h>
#include <algorithm>

NAMESPACE_BEGIN(krr)

template <typename T>
struct Complex {
	KRR_CALLABLE Complex(T re) : re(re), im(0) {}
	KRR_CALLABLE Complex(T re, T im) : re(re), im(im) {}

	KRR_CALLABLE Complex operator-() const { return { -re, -im }; }

	KRR_CALLABLE Complex operator+(Complex z) const { return { re + z.re, im + z.im }; }

	KRR_CALLABLE Complex operator-(Complex z) const { return { re - z.re, im - z.im }; }

	KRR_CALLABLE Complex operator*(Complex z) const {
		return { re * z.re - im * z.im, re * z.im + im * z.re };
	}

	KRR_CALLABLE Complex operator/(Complex z) const {
		T scale = 1 / (z.re * z.re + z.im * z.im);
		return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
	}

	friend KRR_CALLABLE Complex operator+(T value, Complex z) {
		return Complex(value) + z;
	}

	friend KRR_CALLABLE Complex operator-(T value, Complex z) {
		return Complex(value) - z;
	}

	friend KRR_CALLABLE Complex operator*(T value, Complex z) {
		return Complex(value) * z;
	}

	friend KRR_CALLABLE Complex operator/(T value, Complex z) {
		return Complex(value) / z;
	}

	KRR_CALLABLE T real() const { return re; }

	KRR_CALLABLE T imag() const { return im; }

	KRR_CALLABLE T norm() const  { return re * re + im * im; }

	KRR_CALLABLE T abs() const { return std::sqrt(norm()); }

	KRR_CALLABLE Complex<T> sqrt() const {
		T n = this->abs(), t1 = std::sqrt(T(.5) * (n + std::abs(re))), t2 = T(.5) * im / t1;
		if (n == 0) return 0;
		if (re >= 0) return {t1, t2};
		else return {std::abs(t2), std::copysign(t1, im)};
	}

	T re, im;
};

template <typename T> KRR_CALLABLE T real(const Complex<T> &z) { return z.re; }

template <typename T> KRR_CALLABLE T imag(const Complex<T> &z) { return z.im; }

template <typename T> KRR_CALLABLE T norm(const Complex<T> &z) { return z.norm(); }

template <typename T> KRR_CALLABLE T abs(const Complex<T> &z) { return z.abs(); }

template <typename T> KRR_CALLABLE Complex<T> sqrt(const Complex<T> &z) { return z.sqrt(); }

using Complexf = Complex<float>;
using Complexd = Complex<double>;

NAMESPACE_END(krr)