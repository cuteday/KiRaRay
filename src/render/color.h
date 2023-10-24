#pragma once

#include "common.h"
#include "krrmath/math.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

static constexpr int nSpectrumSamples = 4;
constexpr float cLambdaMin = 360, cLambdaMax = 830;

class RGB : public Array<float, 3> {
public:
	using Array::Array;
};

class XYZ : public Array<float, 3> {
public:
	using Array::Array;
};

class RGBSigmoidPolynomial {
public:
	RGBSigmoidPolynomial() = default;

	KRR_CALLABLE RGBSigmoidPolynomial(float c0, float c1, float c2) : c0(c0), c1(c1), c2(c2) {}

	float operator()(float lambda) const {
		return sigmoid(utils::evaluatePolynomial(lambda, c2, c1, c0));
	}

	KRR_CALLABLE float maxValue() const {
		float result = max((*this)(cLambdaMin), (*this)(cLambdaMax));
		float lambda = -c1 / (2 * c0);
		if (lambda >= cLambdaMin && lambda <= cLambdaMax)
			result = max(result, (*this)(lambda));
		return result;
	}

private:
	KRR_CALLABLE static float sigmoid(float x) {
		if (isinf(x)) return x > 0 ? 1 : 0;
		return .5f + x / (2 * sqrt(1 + x * x));
	}

	float c0, c1, c2;
};

KRR_NAMESPACE_END