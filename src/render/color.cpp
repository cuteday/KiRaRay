#include "color.h"
#include "device/gpustd.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

RGBSigmoidPolynomial RGBToSpectrumTable::operator()(RGB rgb) const {
	DCHECK(rgb[0] >= 0.f && rgb[1] >= 0.f && rgb[2] >= 0.f && rgb[0] <= 1.f && rgb[1] <= 1.f &&
		   rgb[2] <= 1.f);

	// Handle uniform _rgb_ values
	if (rgb[0] == rgb[1] && rgb[1] == rgb[2])
		return RGBSigmoidPolynomial(0, 0, (rgb[0] - .5f) / std::sqrt(rgb[0] * (1 - rgb[0])));

	// Find maximum component and compute remapped component values
	int maxc = (rgb[0] > rgb[1]) ? ((rgb[0] > rgb[2]) ? 0 : 2) : ((rgb[1] > rgb[2]) ? 1 : 2);
	float z	 = rgb[maxc];
	float x	 = rgb[(maxc + 1) % 3] * (res - 1) / z;
	float y	 = rgb[(maxc + 2) % 3] * (res - 1) / z;

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
			dz, lerp(dy, lerp(dx, co(0, 0, 0), co(1, 0, 0)), lerp(dx, co(0, 1, 0), co(1, 1, 0))),
			lerp(dy, lerp(dx, co(0, 0, 1), co(1, 0, 1)), lerp(dx, co(0, 1, 1), co(1, 1, 1))));
	}

	return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

extern const int sRGBToSpectrumTable_Res;
extern const float sRGBToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray sRGBToSpectrumTable_Data;
const RGBToSpectrumTable *RGBToSpectrumTable::sRGB;

extern const int DCI_P3ToSpectrumTable_Res;
extern const float DCI_P3ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray DCI_P3ToSpectrumTable_Data;
const RGBToSpectrumTable *RGBToSpectrumTable::DCI_P3;

extern const int REC2020ToSpectrumTable_Res;
extern const float REC2020ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray REC2020ToSpectrumTable_Data;
const RGBToSpectrumTable *RGBToSpectrumTable::Rec2020;

extern const int ACES2065_1ToSpectrumTable_Res;
extern const float ACES2065_1ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray ACES2065_1ToSpectrumTable_Data;
const RGBToSpectrumTable *RGBToSpectrumTable::ACES2065_1;

void RGBToSpectrumTable::init(Allocator alloc) {
	// sRGB
	float *sRGBToSpectrumTableScalePtr =
		(float *) alloc.allocate_bytes(sizeof(sRGBToSpectrumTable_Scale));
	memcpy(sRGBToSpectrumTableScalePtr, sRGBToSpectrumTable_Scale,
			sizeof(sRGBToSpectrumTable_Scale));
	RGBToSpectrumTable::CoefficientArray *sRGBToSpectrumTableDataPtr =
		(RGBToSpectrumTable::CoefficientArray *) alloc.allocate_bytes(
			sizeof(RGBToSpectrumTable::CoefficientArray));
	memcpy(sRGBToSpectrumTableDataPtr, sRGBToSpectrumTable_Data,
			sizeof(sRGBToSpectrumTable_Data));

	sRGB = alloc.new_object<RGBToSpectrumTable>(sRGBToSpectrumTableScalePtr,
												sRGBToSpectrumTableDataPtr);

	// DCI_P3
	float *DCI_P3ToSpectrumTableScalePtr =
		(float *) alloc.allocate_bytes(sizeof(DCI_P3ToSpectrumTable_Scale));
	memcpy(DCI_P3ToSpectrumTableScalePtr, DCI_P3ToSpectrumTable_Scale,
			sizeof(DCI_P3ToSpectrumTable_Scale));
	RGBToSpectrumTable::CoefficientArray *DCI_P3ToSpectrumTableDataPtr =
		(RGBToSpectrumTable::CoefficientArray *) alloc.allocate_bytes(
			sizeof(DCI_P3ToSpectrumTable_Data));
	memcpy(DCI_P3ToSpectrumTableDataPtr, DCI_P3ToSpectrumTable_Data,
			sizeof(DCI_P3ToSpectrumTable_Data));

	DCI_P3 = alloc.new_object<RGBToSpectrumTable>(DCI_P3ToSpectrumTableScalePtr,
													DCI_P3ToSpectrumTableDataPtr);

	// Rec2020
	float *REC2020ToSpectrumTableScalePtr =
		(float *) alloc.allocate_bytes(sizeof(REC2020ToSpectrumTable_Scale));
	memcpy(REC2020ToSpectrumTableScalePtr, REC2020ToSpectrumTable_Scale,
			sizeof(REC2020ToSpectrumTable_Scale));
	RGBToSpectrumTable::CoefficientArray *REC2020ToSpectrumTableDataPtr =
		(RGBToSpectrumTable::CoefficientArray *) alloc.allocate_bytes(
			sizeof(REC2020ToSpectrumTable_Data));
	memcpy(REC2020ToSpectrumTableDataPtr, REC2020ToSpectrumTable_Data,
			sizeof(REC2020ToSpectrumTable_Data));

	Rec2020 = alloc.new_object<RGBToSpectrumTable>(REC2020ToSpectrumTableScalePtr,
													REC2020ToSpectrumTableDataPtr);

	// ACES2065_1
	float *ACES2065_1ToSpectrumTableScalePtr =
		(float *) alloc.allocate_bytes(sizeof(ACES2065_1ToSpectrumTable_Scale));
	memcpy(ACES2065_1ToSpectrumTableScalePtr, ACES2065_1ToSpectrumTable_Scale,
			sizeof(ACES2065_1ToSpectrumTable_Scale));
	RGBToSpectrumTable::CoefficientArray *ACES2065_1ToSpectrumTableDataPtr =
		(RGBToSpectrumTable::CoefficientArray *) alloc.allocate_bytes(
			sizeof(ACES2065_1ToSpectrumTable_Data));
	memcpy(ACES2065_1ToSpectrumTableDataPtr, ACES2065_1ToSpectrumTable_Data,
			sizeof(ACES2065_1ToSpectrumTable_Data));

	ACES2065_1 = alloc.new_object<RGBToSpectrumTable>(ACES2065_1ToSpectrumTableScalePtr,
														ACES2065_1ToSpectrumTableDataPtr);
	return;
}

KRR_NAMESPACE_END