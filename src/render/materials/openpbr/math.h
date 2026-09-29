#pragma once

// Adapted from Adobe OpenPBR BSDF; see NOTICE and LICENSE in this directory.
#include "common.h"
#include "render/sampling.h"
#include "render/spectrum.h"

NAMESPACE_BEGIN(krr)
namespace openpbr {

static KRR_DEVICE_CONST uint16_t idealDielectric[32768] = {
#include "data/openpbr_ideal_dielectric_energy_complement_data.h"
};
static KRR_DEVICE_CONST uint16_t averageDielectric[1024] = {
#include "data/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};
static KRR_DEVICE_CONST uint16_t reflectionRatio[1024] = {
#include "data/openpbr_ideal_dielectric_reflection_ratio_data.h"
};
static KRR_DEVICE_CONST uint16_t opaqueDielectric[32768] = {
#include "data/openpbr_opaque_dielectric_energy_complement_data.h"
};
static KRR_DEVICE_CONST uint16_t averageOpaque[1024] = {
#include "data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};
static KRR_DEVICE_CONST uint16_t idealMetal[1024] = {
#include "data/openpbr_ideal_metal_energy_complement_data.h"
};
static KRR_DEVICE_CONST uint16_t averageMetal[32] = {
#include "data/openpbr_ideal_metal_avg_energy_complement_data.h"
};
#define vec3(x, y, z) {float(x), float(y), float(z)}
static KRR_DEVICE_CONST float fuzzTable[1024][3] = {
#include "data/openpbr_ltc_data.h"
};
#undef vec3

KRR_CALLABLE float square(float x) { return x * x; }
KRR_CALLABLE float pow5(float x) { return square(square(x)) * x; }
KRR_CALLABLE float saturate(float x) { return fminf(fmaxf(x, 0), 1); }
KRR_CALLABLE Spectrum bounded(Spectrum x) { return x.cwiseMax(0.f).cwiseMin(1.f); }
KRR_CALLABLE float fresnel(float eta, float mu) {
	mu = saturate(fabsf(mu));
	if (eta == 1) return 0;
	float sin2 = (1 - square(mu)) / square(eta);
	if (sin2 >= 1) return 1;
	float ct = sqrtf(1 - sin2);
	return .5f *
		   (square((mu - eta * ct) / (mu + eta * ct)) + square((ct - eta * mu) / (ct + eta * mu)));
}
KRR_CALLABLE float averageFresnel(float eta) {
	return eta > 1
			   ? (eta - 1) / (4.08567f + 1.00071f * eta)
			   : .997118f + .1014f * eta - .965241f * square(eta) - .130607f * square(eta) * eta;
}
KRR_CALLABLE float weightedIor(float eta, float weight) {
	float f0 = fminf(.9999f, square((eta - 1) / (eta + 1)) * weight);
	float r = sqrtf(f0), result = (1 + r) / (1 - r);
	return eta < 1 ? 1 / result : result;
}
KRR_CALLABLE float iorIndex(float eta) {
	return fminf(31, fmaxf(0, eta < 1 ? 15 - (1 / eta - 1) * 10 : 16 + (eta - 1) * 10));
}
KRR_CALLABLE float lut1(const uint16_t *data, float x) {
	x	  = fminf(31, fmaxf(0, x));
	int a = int(x), b = a < 31 ? a + 1 : a;
	return (data[a] + (data[b] - float(data[a])) * (x - a)) / 65535.f;
}
KRR_CALLABLE float lut2(const uint16_t *data, float x, float y) {
	x	  = fminf(31, fmaxf(0, x));
	int a = int(x), b = a < 31 ? a + 1 : a;
	return lerp(lut1(data + a * 32, y), lut1(data + b * 32, y), x - a);
}
KRR_CALLABLE float lut3(const uint16_t *data, float x, float y, float z) {
	x	  = fminf(31, fmaxf(0, x));
	int a = int(x), b = a < 31 ? a + 1 : a;
	return lerp(lut2(data + a * 1024, y, z), lut2(data + b * 1024, y, z), x - a);
}
KRR_CALLABLE float extrapolate(float value, float eta) {
	if (eta >= .4f && eta <= 2.5f) return value;
	float f0 = square((eta - 1) / (eta + 1)), edge = square(1.5f / 3.5f);
	return value * (1 - (f0 - edge) / (1 - edge));
}
KRR_CALLABLE float opaqueEnergy(float eta, float alpha, float mu) {
	return extrapolate(lut3(opaqueDielectric, iorIndex(eta), sqrtf(alpha) * 31, mu * 31), eta);
}
KRR_CALLABLE float opaqueAverage(float eta, float alpha) {
	return extrapolate(lut2(averageOpaque, iorIndex(eta), sqrtf(alpha) * 31), eta);
}
KRR_CALLABLE Vector3f fuzzCoefficients(float roughness, float mu) {
	float x = saturate(roughness) * 31, y = saturate(mu) * 31;
	int a = int(x), b = int(y), c = a < 31 ? a + 1 : a, d = b < 31 ? b + 1 : b;
	Vector3f result;
	for (int k = 0; k < 3; ++k)
		result[k] = lerp(lerp(fuzzTable[a * 32 + b][k], fuzzTable[a * 32 + d][k], y - b),
						 lerp(fuzzTable[c * 32 + b][k], fuzzTable[c * 32 + d][k], y - b), x - a);
	return result;
}
KRR_CALLABLE Vector3f fuzzRotate(Vector3f wi, Vector3f wo, bool inverse = false) {
	float length = sqrtf(square(wo[0]) + square(wo[1]));
	if (length == 0) return wi;
	float c = wo[0] / length, s = wo[1] / length * (inverse ? -1 : 1);
	return {c * wi[0] + s * wi[1], -s * wi[0] + c * wi[1], wi[2]};
}
KRR_CALLABLE float fuzzPdf(Vector3f wi, Vector3f wo, Vector3f coefficients) {
	wi = fuzzRotate(wi, wo);
	if (wi[2] <= 0 || coefficients[0] <= 0) return 0;
	Vector3f v(coefficients[0] * wi[0] + coefficients[1] * wi[2], coefficients[0] * wi[1], wi[2]);
	float length2 = v.squaredNorm();
	return length2 > 0 ? M_INV_PI * wi[2] * square(coefficients[0] / length2) : 0;
}
KRR_CALLABLE Vector3f sampleFuzz(Vector2f u, Vector3f wo, Vector3f coefficients) {
	if (coefficients[0] <= 0) return Vector3f(0.f);
	Vector3f q = cosineSampleHemisphere(u);
	return fuzzRotate(normalize(Vector3f((q[0] - coefficients[1] * q[2]) / coefficients[0],
										 q[1] / coefficients[0], q[2])),
					  wo, true);
}

struct GGX {
	float ax, ay;
	KRR_CALLABLE GGX(float alpha, float anisotropy = 0) {
		float ratio = 1 - saturate(anisotropy);
		ax			= fmaxf(1e-6f, alpha * sqrtf(2 / (1 + square(ratio))));
		ay			= fmaxf(1e-6f, ratio * ax);
	}
	KRR_CALLABLE float D(Vector3f m) const {
		float d = square(m[0] / ax) + square(m[1] / ay) + square(m[2]);
		return m[2] > 0 && d > 0 ? 1 / (M_PI * ax * ay * square(d)) : 0;
	}
	KRR_CALLABLE float G1(Vector3f v) const {
		if (v[2] == 0) return 0;
		return 2 / (1 + sqrtf(1 + (square(ax * v[0]) + square(ay * v[1])) / square(v[2])));
	}
	KRR_CALLABLE float normalPdf(Vector3f wo, Vector3f m) const {
		return wo[2] > 0 ? D(m) * G1(wo) * fmaxf(0, dot(wo, m)) / wo[2] : 0;
	}
	KRR_CALLABLE float reflectionPdf(Vector3f wo, Vector3f wi) const {
		if (wo[2] <= 0 || wi[2] <= 0) return 0;
		Vector3f m = normalize(wo + wi);
		return reflectionPdf(wo, wi, m);
	}
	KRR_CALLABLE float reflectionPdf(Vector3f wo, Vector3f wi, Vector3f m) const {
		if (wo[2] <= 0 || wi[2] <= 0) return 0;
		return D(m) * G1(wo) / (4 * wo[2]);
	}
	KRR_CALLABLE float reflectionCos(Vector3f wo, Vector3f wi) const {
		return reflectionPdf(wo, wi) * G1(wi);
	}
	KRR_CALLABLE Vector3f sampleNormal(Vector3f wo, Vector2f u) const {
		Vector3f v = normalize(Vector3f(ax * wo[0], ay * wo[1], wo[2]));
		float phi = 2 * M_PI * u[0], z = (1 - u[1]) * (1 + v[2]);
		float r = sqrtf(fmaxf(0, u[1] * (1 + v[2]) * (z + 1 - v[2])));
		return normalize(Vector3f(ax * (r * cosf(phi) + v[0]), ay * (r * sinf(phi) + v[1]), z));
	}
};

KRR_CALLABLE Spectrum metalB(Spectrum f0, Spectrum tint) {
	return ((f0 + (Spectrum(1) - f0) * pow5(6.f / 7)) * (Spectrum(1) - tint)) /
		   ((1.f / 7) * pow5(6.f / 7) * (6.f / 7));
}
KRR_CALLABLE Spectrum metalFresnel(Spectrum f0, Spectrum tint, float mu) {
	return bounded(f0 + ((Spectrum(1) - f0) - metalB(f0, tint) * mu * (1 - mu)) * pow5(1 - mu));
}
KRR_CALLABLE Spectrum metalAverage(Spectrum f0, Spectrum tint) {
	return bounded(f0 + (Spectrum(1) - f0) / 21 - metalB(f0, tint) / 126);
}
KRR_CALLABLE float diffuseEnergy(float mu, float roughness) {
	float x = 1 - mu;
	float g = x * (.0571085289f + x * (.491881867f + x * (-.332181442f + x * .0714429953f)));
	return (1 + roughness * g) / (1 + (.5f - 2.f / (3 * M_PI)) * roughness);
}
KRR_CALLABLE Spectrum diffuse(Spectrum rho, float roughness, Vector3f wo, Vector3f wi) {
	float a			   = 1 / (1 + (.5f - 2.f / (3 * M_PI)) * roughness);
	float s			   = dot(wi, wo) - wi[2] * wo[2];
	float st		   = s > 0 ? s / fmaxf(wi[2], wo[2]) : s;
	float average	   = a * (1 + (2.f / 3 - 28.f / (15 * M_PI)) * roughness);
	Spectrum ms		   = rho * rho * average / (Spectrum(1) - rho * (1 - average));
	float compensation = fmaxf(1e-7f, 1 - diffuseEnergy(wo[2], roughness)) *
						 fmaxf(1e-7f, 1 - diffuseEnergy(wi[2], roughness)) /
						 fmaxf(1e-7f, 1 - average);
	return M_INV_PI * (rho * (a * (1 + roughness * st)) + ms * compensation);
}

} // namespace openpbr
NAMESPACE_END(krr)
