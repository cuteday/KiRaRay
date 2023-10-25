#pragma once

#include "common.h"
#include "krrmath/math.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

#ifdef RGB
#undef RGB
#endif

static constexpr int nSpectrumSamples = 4;
constexpr float cLambdaMin = 360, cLambdaMax = 830;

class RGB : public Array3f {
public:
	using Array3f::Array3f;
};

class XYZ : public Array3f {
public:
	using Array3f::Array3f;
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

class RGBToSpectrumTable {
public:
	// RGBToSpectrumTable Public Constants
	static constexpr int res = 64;

	using CoefficientArray = float[3][res][res][res][3];

	// RGBToSpectrumTable Public Methods
	KRR_CALLABLE RGBToSpectrumTable(const float *zNodes, const CoefficientArray *coeffs) :
		zNodes(zNodes), coeffs(coeffs) {}

	KRR_CALLABLE RGBSigmoidPolynomial operator()(RGB rgb) const;

	static void init(Allocator alloc);

	static const RGBToSpectrumTable *sRGB;
	static const RGBToSpectrumTable *DCI_P3;
	static const RGBToSpectrumTable *Rec2020;
	static const RGBToSpectrumTable *ACES2065_1;

private:
	// RGBToSpectrumTable Private Members
	const float *zNodes;
	const CoefficientArray *coeffs;
};

KRR_NAMESPACE_END