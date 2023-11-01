#include "spectrum.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

namespace {
#include "data/named_spectrum.cpp"
} // anonymous namespace

float SampledSpectrum::y(const SampledWavelengths &lambda) const {
	SampledSpectrum Ys	= spec::Y().sample(lambda);
	SampledSpectrum pdf = lambda.pdf();
	return SampledSpectrum(Ys * *this).safeDiv(pdf).mean() / CIE_Y_integral;
}

XYZ SampledSpectrum::toXYZ(const SampledWavelengths &lambda) const {
	// Sample the $X$, $Y$, and $Z$ matching curves at _lambda_
	SampledSpectrum X = spec::X().sample(lambda);
	SampledSpectrum Y = spec::Y().sample(lambda);
	SampledSpectrum Z = spec::Z().sample(lambda);
	// Evaluate estimator to compute $(x,y,z)$ coefficients
	SampledSpectrum pdf = lambda.pdf();
	return XYZ(SampledSpectrum(X * *this).safeDiv(pdf).mean(),
			   SampledSpectrum(Y * *this).safeDiv(pdf).mean(),
			   SampledSpectrum(Z * *this).safeDiv(pdf).mean()) /
		   CIE_Y_integral;
}

RGB SampledSpectrum::toRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const {
	XYZ xyz = toXYZ(lambda);
	return cs.toRGB(xyz);
}

float PiecewiseLinearSpectrum::operator()(float lambda) const {
	// Handle _PiecewiseLinearSpectrum_ corner cases
	if (lambdas.empty() || lambda < lambdas.front() || lambda > lambdas.back()) return 0;
	// Find offset to largest _lambdas_ below _lambda_ and interpolate
	int o = utils::findInterval(lambdas.size(), [&](int i) { return lambdas[i] <= lambda; });
	DCHECK(lambda >= lambdas[o] && lambda <= lambdas[o + 1]);
	float t = (lambda - lambdas[o]) / (lambdas[o + 1] - lambdas[o]);
	return lerp(values[o], values[o + 1], t);
}

PiecewiseLinearSpectrum::PiecewiseLinearSpectrum(gpu::span<const float> l, gpu::span<const float> v,
												 Allocator alloc) :
	lambdas(l.begin(), l.end(), alloc), values(v.begin(), v.end(), alloc) {
	CHECK_EQ(lambdas.size(), values.size());
	for (size_t i = 0; i < lambdas.size() - 1; ++i) CHECK_LT(lambdas[i], lambdas[i + 1]);
}

PiecewiseLinearSpectrum *PiecewiseLinearSpectrum::fromInterleaved(gpu::span<const float> samples,
																  bool normalize, Allocator alloc) {
	CHECK_EQ(0, samples.size() % 2);
	int n = samples.size() / 2;
	gpu::vector<float> lambda, v;

	// Extend samples to cover range of visible wavelengths if needed.
	if (samples[0] > cLambdaMin) {
		lambda.push_back(cLambdaMin - 1);
		v.push_back(samples[1]);
	}
	for (size_t i = 0; i < n; ++i) {
		lambda.push_back(samples[2 * i]);
		v.push_back(samples[2 * i + 1]);
		if (i > 0) CHECK_GT(lambda.back(), lambda[lambda.size() - 2]);
	}
	if (lambda.back() < cLambdaMax) {
		lambda.push_back(cLambdaMax + 1);
		v.push_back(v.back());
	}

	PiecewiseLinearSpectrum *spec =
		alloc.new_object<PiecewiseLinearSpectrum>(lambda, v, alloc);

	if (normalize)
		// Normalize to have luminance of 1.
		spec->scale(CIE_Y_integral / spec::innerProduct(spec, &spec::Y()));

	return spec;
}

const RGBColorSpace *RGBColorSpace::getNamed(std::string n) {
	std::string name;
	std::transform(n.begin(), n.end(), std::back_inserter(name), ::tolower);
	if (name == "aces2065-1") return ACES2065_1;
	else if (name == "rec2020") return Rec2020;
	else if (name == "dci-p3") return DCI_P3;
	else if (name == "srgb") return sRGB;
	else return nullptr;
}

const RGBColorSpace *RGBColorSpace::lookup(Point2f r, Point2f g, Point2f b, Point2f w) {
	auto closeEnough = [](const Point2f &a, const Point2f &b) {
		return ((a.x() == b.x() || std::abs((a.x() - b.x()) / b.x()) < 1e-3) &&
				(a.y() == b.y() || std::abs((a.y() - b.y()) / b.y()) < 1e-3));
	};
	for (const RGBColorSpace *cs : {ACES2065_1, DCI_P3, Rec2020, sRGB}) {
		if (closeEnough(r, cs->r) && closeEnough(g, cs->g) && closeEnough(b, cs->b) &&
			closeEnough(w, cs->w))
			return cs;
	}
	return nullptr;
}

const RGBColorSpace *RGBColorSpace::sRGB;
const RGBColorSpace *RGBColorSpace::DCI_P3;
const RGBColorSpace *RGBColorSpace::Rec2020;
const RGBColorSpace *RGBColorSpace::ACES2065_1;

namespace spec {

std::map<std::string, Spectrum> namedSpectra;

KRR_DEVICE DenselySampledSpectrum *xGPU, *yGPU, *zGPU;
DenselySampledSpectrum *x, *y, *z;

KRR_CONST RGBColorSpace *RGBColorSpace_sRGB;
KRR_CONST RGBColorSpace *RGBColorSpace_DCI_P3;
KRR_CONST RGBColorSpace *RGBColorSpace_Rec2020;
KRR_CONST RGBColorSpace *RGBColorSpace_ACES2065_1;

void init(Allocator alloc) {
	PiecewiseLinearSpectrum xpls(CIE_lambda, CIE_X);
	x = alloc.new_object<DenselySampledSpectrum>(&xpls, alloc);

	PiecewiseLinearSpectrum ypls(CIE_lambda, CIE_Y);
	y = alloc.new_object<DenselySampledSpectrum>(&ypls, alloc);

	PiecewiseLinearSpectrum zpls(CIE_lambda, CIE_Z);
	z = alloc.new_object<DenselySampledSpectrum>(&zpls, alloc);

	CUDA_CHECK(cudaMemcpyToSymbol(xGPU, &x, sizeof(x)));
	CUDA_CHECK(cudaMemcpyToSymbol(yGPU, &y, sizeof(y)));
	CUDA_CHECK(cudaMemcpyToSymbol(zGPU, &z, sizeof(z)));

	Spectrum illuma		  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_A, true, alloc);
	Spectrum illumd50	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_D5000, true, alloc);
	Spectrum illumacesd60 = PiecewiseLinearSpectrum::fromInterleaved(ACES_Illum_D60, true, alloc);
	Spectrum illumd65	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_D6500, true, alloc);
	Spectrum illumf1	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F1, true, alloc);
	Spectrum illumf2	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F2, true, alloc);
	Spectrum illumf3	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F3, true, alloc);
	Spectrum illumf4	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F4, true, alloc);
	Spectrum illumf5	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F5, true, alloc);
	Spectrum illumf6	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F6, true, alloc);
	Spectrum illumf7	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F7, true, alloc);
	Spectrum illumf8	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F8, true, alloc);
	Spectrum illumf9	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F9, true, alloc);
	Spectrum illumf10	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F10, true, alloc);
	Spectrum illumf11	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F11, true, alloc);
	Spectrum illumf12	  = PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F12, true, alloc);

	Spectrum ageta		   = PiecewiseLinearSpectrum::fromInterleaved(Ag_eta, false, alloc);
	Spectrum agk		   = PiecewiseLinearSpectrum::fromInterleaved(Ag_k, false, alloc);
	Spectrum aleta		   = PiecewiseLinearSpectrum::fromInterleaved(Al_eta, false, alloc);
	Spectrum alk		   = PiecewiseLinearSpectrum::fromInterleaved(Al_k, false, alloc);
	Spectrum aueta		   = PiecewiseLinearSpectrum::fromInterleaved(Au_eta, false, alloc);
	Spectrum auk		   = PiecewiseLinearSpectrum::fromInterleaved(Au_k, false, alloc);
	Spectrum cueta		   = PiecewiseLinearSpectrum::fromInterleaved(Cu_eta, false, alloc);
	Spectrum cuk		   = PiecewiseLinearSpectrum::fromInterleaved(Cu_k, false, alloc);
	Spectrum cuzneta	   = PiecewiseLinearSpectrum::fromInterleaved(CuZn_eta, false, alloc);
	Spectrum cuznk		   = PiecewiseLinearSpectrum::fromInterleaved(CuZn_k, false, alloc);
	Spectrum mgoeta		   = PiecewiseLinearSpectrum::fromInterleaved(MgO_eta, false, alloc);
	Spectrum mgok		   = PiecewiseLinearSpectrum::fromInterleaved(MgO_k, false, alloc);
	Spectrum tio2eta	   = PiecewiseLinearSpectrum::fromInterleaved(TiO2_eta, false, alloc);
	Spectrum tio2k		   = PiecewiseLinearSpectrum::fromInterleaved(TiO2_k, false, alloc);
	Spectrum glassbk7eta   = PiecewiseLinearSpectrum::fromInterleaved(GlassBK7_eta, false, alloc);
	Spectrum glassbaf10eta = PiecewiseLinearSpectrum::fromInterleaved(GlassBAF10_eta, false, alloc);
	Spectrum glassfk51aeta = PiecewiseLinearSpectrum::fromInterleaved(GlassFK51A_eta, false, alloc);
	Spectrum glasslasf9eta = PiecewiseLinearSpectrum::fromInterleaved(GlassLASF9_eta, false, alloc);
	Spectrum glasssf5eta   = PiecewiseLinearSpectrum::fromInterleaved(GlassSF5_eta, false, alloc);
	Spectrum glasssf10eta  = PiecewiseLinearSpectrum::fromInterleaved(GlassSF10_eta, false, alloc);
	Spectrum glasssf11eta  = PiecewiseLinearSpectrum::fromInterleaved(GlassSF11_eta, false, alloc);

	namedSpectra = {
		{"glass-BK7", glassbk7eta},
		{"glass-BAF10", glassbaf10eta},
		{"glass-FK51A", glassfk51aeta},
		{"glass-LASF9", glasslasf9eta},
		{"glass-F5", glasssf5eta},
		{"glass-F10", glasssf10eta},
		{"glass-F11", glasssf11eta},

		{"metal-Ag-eta", ageta},
		{"metal-Ag-k", agk},
		{"metal-Al-eta", aleta},
		{"metal-Al-k", alk},
		{"metal-Au-eta", aueta},
		{"metal-Au-k", auk},
		{"metal-Cu-eta", cueta},
		{"metal-Cu-k", cuk},
		{"metal-CuZn-eta", cuzneta},
		{"metal-CuZn-k", cuznk},
		{"metal-MgO-eta", mgoeta},
		{"metal-MgO-k", mgok},
		{"metal-TiO2-eta", tio2eta},
		{"metal-TiO2-k", tio2k},

		{"stdillum-A", illuma},
		{"stdillum-D50", illumd50},
		{"stdillum-D65", illumd65},
		{"stdillum-F1", illumf1},
		{"stdillum-F2", illumf2},
		{"stdillum-F3", illumf3},
		{"stdillum-F4", illumf4},
		{"stdillum-F5", illumf5},
		{"stdillum-F6", illumf6},
		{"stdillum-F7", illumf7},
		{"stdillum-F8", illumf8},
		{"stdillum-F9", illumf9},
		{"stdillum-F10", illumf10},
		{"stdillum-F11", illumf11},
		{"stdillum-F12", illumf12},

		{"illum-acesD60", illumacesd60},

		{"canon_eos_100d_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_r, false, alloc)},
		{"canon_eos_100d_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_g, false, alloc)},
		{"canon_eos_100d_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_b, false, alloc)},

		{"canon_eos_1dx_mkii_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_r, false, alloc)},
		{"canon_eos_1dx_mkii_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_g, false, alloc)},
		{"canon_eos_1dx_mkii_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_b, false, alloc)},

		{"canon_eos_200d_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_r, false, alloc)},
		{"canon_eos_200d_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_g, false, alloc)},
		{"canon_eos_200d_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_b, false, alloc)},

		{"canon_eos_200d_mkii_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_r, false, alloc)},
		{"canon_eos_200d_mkii_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_g, false, alloc)},
		{"canon_eos_200d_mkii_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_b, false, alloc)},

		{"canon_eos_5d_r", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_r, false, alloc)},
		{"canon_eos_5d_g", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_g, false, alloc)},
		{"canon_eos_5d_b", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_b, false, alloc)},

		{"canon_eos_5d_mkii_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_r, false, alloc)},
		{"canon_eos_5d_mkii_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_g, false, alloc)},
		{"canon_eos_5d_mkii_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_b, false, alloc)},

		{"canon_eos_5d_mkiii_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_r, false, alloc)},
		{"canon_eos_5d_mkiii_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_g, false, alloc)},
		{"canon_eos_5d_mkiii_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_b, false, alloc)},

		{"canon_eos_5d_mkiv_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_r, false, alloc)},
		{"canon_eos_5d_mkiv_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_g, false, alloc)},
		{"canon_eos_5d_mkiv_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_b, false, alloc)},

		{"canon_eos_5ds_r",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_r, false, alloc)},
		{"canon_eos_5ds_g",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_g, false, alloc)},
		{"canon_eos_5ds_b",
		 PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_b, false, alloc)},

		{"canon_eos_m_r", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_r, false, alloc)},
		{"canon_eos_m_g", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_g, false, alloc)},
		{"canon_eos_m_b", PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_b, false, alloc)},

		{"hasselblad_l1d_20c_r",
		 PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_r, false, alloc)},
		{"hasselblad_l1d_20c_g",
		 PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_g, false, alloc)},
		{"hasselblad_l1d_20c_b",
		 PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_b, false, alloc)},

		{"nikon_d810_r", PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_r, false, alloc)},
		{"nikon_d810_g", PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_g, false, alloc)},
		{"nikon_d810_b", PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_b, false, alloc)},

		{"nikon_d850_r", PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_r, false, alloc)},
		{"nikon_d850_g", PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_g, false, alloc)},
		{"nikon_d850_b", PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_b, false, alloc)},

		{"sony_ilce_6400_r",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_r, false, alloc)},
		{"sony_ilce_6400_g",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_g, false, alloc)},
		{"sony_ilce_6400_b",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_b, false, alloc)},

		{"sony_ilce_7m3_r",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_r, false, alloc)},
		{"sony_ilce_7m3_g",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_g, false, alloc)},
		{"sony_ilce_7m3_b",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_b, false, alloc)},

		{"sony_ilce_7rm3_r",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_r, false, alloc)},
		{"sony_ilce_7rm3_g",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_g, false, alloc)},
		{"sony_ilce_7rm3_b",
		 PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_b, false, alloc)},

		{"sony_ilce_9_r", PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_r, false, alloc)},
		{"sony_ilce_9_g", PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_g, false, alloc)},
		{"sony_ilce_9_b", PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_b, false, alloc)}};
}

float innerProduct(Spectrum f, Spectrum g) {
	float integral = 0;
	for (float lambda = cLambdaMin; lambda <= cLambdaMax; ++lambda)
		integral += f(lambda) * g(lambda);
	return integral;
}

XYZ spectrumToXYZ(Spectrum s) {
	return XYZ(innerProduct(&X(), s), innerProduct(&Y(), s), innerProduct(&Z(), s)) /
		   CIE_Y_integral;
}

Spectrum getNamedSpectrum(std::string name) {
	auto iter = spec::namedSpectra.find(name);
	if (iter != spec::namedSpectra.end()) return iter->second;
	return nullptr;
}
} // namespace spec

void RGBColorSpace::init(Allocator alloc) {
	using namespace spec;
	// Rec. ITU-R BT.709.3
	sRGB = alloc.new_object<RGBColorSpace>(
		Point2f(.64, .33), Point2f(.3, .6), Point2f(.15, .06), getNamedSpectrum("stdillum-D65"),
		RGBToSpectrumTable::sRGB, spec::x, spec::y, spec::z, alloc);
	// P3-D65 (display)
	DCI_P3 = alloc.new_object<RGBColorSpace>(
		Point2f(.68, .32), Point2f(.265, .690), Point2f(.15, .06), getNamedSpectrum("stdillum-D65"),
		RGBToSpectrumTable::DCI_P3, spec::x, spec::y, spec::z, alloc);
	// ITU-R Rec BT.2020
	Rec2020	   = alloc.new_object<RGBColorSpace>(Point2f(.708, .292), Point2f(.170, .797),
												 Point2f(.131, .046), getNamedSpectrum("stdillum-D65"),
												 RGBToSpectrumTable::Rec2020, spec::x, spec::y,
												 spec::z, alloc);
	ACES2065_1 = alloc.new_object<RGBColorSpace>(
		Point2f(.7347, .2653), Point2f(0., 1.), Point2f(.0001, -.077),
		getNamedSpectrum("illum-acesD60"), RGBToSpectrumTable::ACES2065_1, spec::x, spec::y,
		spec::z, alloc);

	CUDA_CHECK(
		cudaMemcpyToSymbol(RGBColorSpace_sRGB, &RGBColorSpace::sRGB, sizeof(RGBColorSpace_sRGB)));
	CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_DCI_P3, &RGBColorSpace::DCI_P3,
								  sizeof(RGBColorSpace_DCI_P3)));
	CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_Rec2020, &RGBColorSpace::Rec2020,
								  sizeof(RGBColorSpace_Rec2020)));
	CUDA_CHECK(cudaMemcpyToSymbol(RGBColorSpace_ACES2065_1, &RGBColorSpace::ACES2065_1,
								  sizeof(RGBColorSpace_ACES2065_1)));
}

KRR_NAMESPACE_END