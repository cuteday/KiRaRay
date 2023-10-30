#include "render/color.h"
#include "render/spectrum.h"
#include "device/gpustd.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN

RGBColorSpace::RGBColorSpace(Point2f r, Point2f g, Point2f b, Spectrum illuminant,
							 const RGBToSpectrumTable *rgbToSpec, Allocator alloc) :
	r(r), g(g), b(b), illuminant(illuminant, alloc), rgbToSpectrumTable(rgbToSpec) {
	// Compute whitepoint primaries and XYZ coordinates
	XYZ W = spec::spectrumToXYZ(illuminant);
	w	  = W.xy();
	XYZ R = XYZ::fromxyY(r), G = XYZ::fromxyY(g), B = XYZ::fromxyY(b);

	// Initialize XYZ color space conversion matrices
	Matrix3f rgb{{R.x(), G.x(), B.x()}, {R.y(), G.y(), B.y()}, {R.z(), G.z(), B.z()}};
	XYZ C	   = rgb.inverse() * Vector3f(W);
	XYZFromRGB = rgb * Vector3f(C).asDiagonal().toDenseMatrix();
	RGBFromXYZ = XYZFromRGB.inverse();
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