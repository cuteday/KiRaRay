#pragma once

#include "common.h"
#include "krrmath/math.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

#ifdef RGB
#undef RGB
#endif

#define CIE_Y_integral				106.856895f
#define KRR_DEFAULT_COLORSPACE		RGBColorSpace::sRGB
#define KRR_DEFAULT_COLORSPACE_GPU  spec::RGBColorSpace_sRGB
constexpr float cLambdaMin = 360, cLambdaMax = 830;
static constexpr int nSpectrumSamples = KRR_RENDER_SPECTRAL ? 4 : 3;

class RGB;
class SampledSpectrum;
class RGBColorSpace;
class SampledWavelengths;

#if KRR_RENDER_SPECTRAL
typedef SampledSpectrum Spectrum;
#else
typedef RGB Spectrum;
#endif

enum class SpectrumType {
	RGBBounded,
	RGBUnbounded,
	RGBIlluminant,
};

class RGB : public Array3f {
public:
	using Array3f::Array3f;
	KRR_CALLABLE float r() const { return (*this)[0]; }
	KRR_CALLABLE float g() const { return (*this)[1]; }
	KRR_CALLABLE float b() const { return (*this)[2]; }

	KRR_CALLABLE static RGB fromRGB(const RGB &rgb, SpectrumType type,
									const SampledWavelengths &lambda,
									const RGBColorSpace &colorSpace);
	KRR_CALLABLE RGB toRGB(const SampledWavelengths &lambda, const RGBColorSpace &colorSpace);
};

class RGBA : public Array4f {
public:
	using Array4f::Array4f;
	KRR_CALLABLE float r() const { return (*this)[0]; }
	KRR_CALLABLE float g() const { return (*this)[1]; }
	KRR_CALLABLE float b() const { return (*this)[2]; }
	KRR_CALLABLE float a() const { return (*this)[3]; }
};

class XYZ : public Array3f {
public:
	using Array3f::Array3f;

	KRR_CALLABLE Point2f xy() const { return {x() / sum(), y() / sum()}; } 

	KRR_CALLABLE static XYZ fromxyY(Point2f xy, float Y = 1) {
		if (xy.y() == 0) return {0, 0, 0};
		return {xy.x() * Y / xy.y(), Y, (1 - xy.x() - xy.y()) * Y / xy.y()};
	}
};

class RGBSigmoidPolynomial {
public:
	RGBSigmoidPolynomial() = default;

	KRR_CALLABLE RGBSigmoidPolynomial(float c0, float c1, float c2) : c0(c0), c1(c1), c2(c2) {}

	KRR_CALLABLE float operator()(float lambda) const {
		return sigmoid(utils::evaluatePolynomial(lambda, c2, c1, c0));
	}

	KRR_CALLABLE float maxValue() const {
		float result = max((*this)(cLambdaMin), (*this)(cLambdaMax));
		float lambda = -c1 / (2 * c0);
		if (lambda >= cLambdaMin && lambda <= cLambdaMax)
			result = max(result, (*this)(lambda));
		return result;
	}

	float c0, c1, c2;
private:
	KRR_CALLABLE static float sigmoid(float x) {
		if (isinf(x)) return x > 0 ? 1 : 0;
		return .5f + x / (2 * sqrt(1 + x * x));
	}
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

inline RGBSigmoidPolynomial RGBToSpectrumTable::operator()(RGB rgb) const {
	DCHECK(rgb[0] >= 0.f && rgb[1] >= 0.f && rgb[2] >= 0.f && rgb[0] <= 1.f && rgb[1] <= 1.f &&
		   rgb[2] <= 1.f);

	// Handle uniform _rgb_ values
	if (rgb[0] == rgb[1] && rgb[1] == rgb[2])
		return RGBSigmoidPolynomial(0, 0, (rgb[0] - .5f) / std::sqrt(rgb[0] * (1 - rgb[0])));

	// Find maximum component and compute remapped component values
	int maxc;
	rgb.maxCoeff(&maxc);
	float z = rgb[maxc];
	float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
	float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

	// Compute integer indices and offsets for coefficient interpolation
	int xi = std::min((int) x, res - 2), yi = std::min((int) y, res - 2),
		zi	 = utils::findInterval(res, [&](int i) { return zNodes[i] < z; });
	float dx = x - xi, dy = y - yi, dz = (z - zNodes[zi]) / (zNodes[zi + 1] - zNodes[zi]);

	// Trilinearly interpolate sigmoid polynomial coefficients _c_
	Array3f c;
	for (int i = 0; i < 3; ++i) {
		// Define _co_ lambda for looking up sigmoid polynomial coefficients
		auto co = [&](int dx, int dy, int dz) {
			return (*coeffs)[maxc][zi + dz][yi + dy][xi + dx][i];
		};

		c[i] = lerp(
			lerp(lerp(co(0, 0, 0), co(1, 0, 0), dx), lerp(co(0, 1, 0), co(1, 1, 0), dx), dy),
			lerp(lerp(co(0, 0, 1), co(1, 0, 1), dx), lerp(co(0, 1, 1), co(1, 1, 1), dx), dy), dz);
	}
	return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

KRR_NAMESPACE_END