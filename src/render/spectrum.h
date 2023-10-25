#pragma once

#include "common.h"
#include "krrmath/math.h"
#include "device/gpustd.h"
#include "device/taggedptr.h"
#include "util/check.h"
#include "render/color.h"

KRR_NAMESPACE_BEGIN

class SampledSpectrum: public Array<float, nSpectrumSamples> {
public:
	SampledSpectrum() = default;
	KRR_CALLABLE explicit SampledSpectrum(float c) : Array(c) {}

	KRR_CALLABLE SampledSpectrum(gpu::span<const float> v) {
		DCHECK_EQ(v.size(), nSpectrumSamples);
		for (int i=0; i<nSpectrumSamples; i++)
			(*this)[i] = v[i];
	}

	KRR_CALLABLE explicit operator bool() const { return this->any(); }
};

class SampledWavelengths {
public:
	KRR_CALLABLE bool operator==(const SampledWavelengths& swl) const {
		return (lambda == swl.lambda).all() && (pdfs == swl.pdfs).all();
	}

	KRR_CALLABLE bool operator!=(const SampledWavelengths& swl) const {
		return !(lambda == swl.lambda).all() || !(pdfs == swl.pdfs).all();
	}

	KRR_CALLABLE static SampledWavelengths sampleUniform(float u, float lambdaMin = cLambdaMin,
		float lambdaMax = cLambdaMax) {
		SampledWavelengths swl;
		swl.lambda[0] = lerp(u, lambdaMin, lambdaMax);
		float delta = (lambdaMax - lambdaMin) / nSpectrumSamples;
		for (int i = 1; i < nSpectrumSamples; i++) {
			swl.lambda[i] = swl.lambda[i - 1] + delta;
			if (swl.lambda[i] > lambdaMax) 
				swl.lambda[i] -= lambdaMax - lambdaMin;
		}
		swl.pdfs.fill(1.f / (lambdaMax - lambdaMin));
		return swl;
	}

	KRR_CALLABLE float operator[](int i) const { return lambda[i]; }
	KRR_CALLABLE float &operator[](int i) { return lambda[i]; }

	Array<float, nSpectrumSamples> pdf() const { return pdfs; }

private:
	Array<float, nSpectrumSamples> lambda, pdfs;
};

class RGBBoundedSpectrum {
public:
	KRR_CALLABLE float operator()(float lambda) const { return rsp(lambda); }

	KRR_CALLABLE float maxValue() const { return rsp.maxValue(); }

	KRR_CALLABLE RGBBoundedSpectrum(RGB rgb) {}

	KRR_CALLABLE SampledSpectrum sample(const SampledWavelengths& lambda) const {
		SampledSpectrum result;
		for (int i = 0; i < nSpectrumSamples; i++)
			result[i] = rsp(lambda[i]);
		return result;
	}

private:
	RGBSigmoidPolynomial rsp;
};

class RGBUnboundedSpectrum {
public:
	KRR_CALLABLE float operator()(float lambda) const { return scale * rsp(lambda); }

	KRR_CALLABLE float maxValue() const { return scale * rsp.maxValue(); }

	KRR_CALLABLE RGBUnboundedSpectrum(): rsp(0, 0, 0), scale(0) {}
	KRR_CALLABLE RGBUnboundedSpectrum(RGB rgb) {}

	KRR_CALLABLE SampledSpectrum sample(const SampledWavelengths &lambda) const {
		SampledSpectrum result;
		for (int i = 0; i < nSpectrumSamples; i++) result[i] = scale * rsp(lambda[i]);
		return result;
	}

private:
	float scale;
	RGBSigmoidPolynomial rsp;
};

class Spectrum : public TaggedPointer<RGBBoundedSpectrum, RGBUnboundedSpectrum> {
public:
	using TaggedPointer::TaggedPointer;
	KRR_CALLABLE float operator()(float lambda) const;
	KRR_CALLABLE float maxValue() const;
	KRR_CALLABLE SampledSpectrum sample(const SampledWavelengths &lambda) const;
};

class RGBColorSpace {
public:
	// RGBColorSpace Public Methods
	RGBColorSpace(Point2f r, Point2f g, Point2f b, Spectrum illuminant,
				  const RGBToSpectrumTable *rgbToSpectrumTable, Allocator alloc);

	KRR_CALLABLE RGBSigmoidPolynomial toRGBCoeffs(RGB rgb) const;

	static void init(Allocator alloc);

	// RGBColorSpace Public Members
	Point2f r, g, b, w;
	DenselySampledSpectrum illuminant;
	Matrix3f XYZFromRGB, RGBFromXYZ;
	static const RGBColorSpace *sRGB, *DCI_P3, *Rec2020, *ACES2065_1;

	KRR_CALLABLE bool operator==(const RGBColorSpace &cs) const {
		return (r == cs.r && g == cs.g && b == cs.b && w == cs.w &&
				rgbToSpectrumTable == cs.rgbToSpectrumTable);
	}
	KRR_CALLABLE bool operator!=(const RGBColorSpace &cs) const {
		return (r != cs.r || g != cs.g || b != cs.b || w != cs.w ||
				rgbToSpectrumTable != cs.rgbToSpectrumTable);
	}

	KRR_CALLABLE RGB luminanceVector() const {
		return RGB(XYZFromRGB(1, 0), XYZFromRGB(1, 1), XYZFromRGB(1, 2));
	}

	static const RGBColorSpace *getNamed(std::string name);
	static const RGBColorSpace *lookup(Point2f r, Point2f g, Point2f b, Point2f w);

private:
	// RGBColorSpace Private Members
	const RGBToSpectrumTable *rgbToSpectrumTable;
};


KRR_NAMESPACE_END