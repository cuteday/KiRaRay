#include <iostream>
#include "render/materials/openpbr/math.h"
#include "material/program.h"

using namespace krr;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

int main() {
	try {
		using namespace openpbr;
		require(materialEmissionIntensity(1000, MaterialModel::OpenPBR) == 1 &&
					materialEmissionIntensity(2, MaterialModel::PreviewSurface) == 2 &&
					materialEmissionIntensity(-1, MaterialModel::OpenPBR) == 0,
				"OpenPBR nits and Preview Surface radiance units");
		require(fabsf(fresnel(1.5f, 1) - .04f) < 1e-6f, "Dielectric normal incidence Fresnel");
		require(fresnel(1, .1f) == 0 && fresnel(1 / 1.5f, .1f) == 1,
				"Index matching and total internal reflection");
		require(fabsf(weightedIor(1.5f, 1) - 1.5f) < 1e-6f && weightedIor(1.5f, 0) == 1,
				"Specular weight modifies IOR");
		Spectrum f0(.4f), tint(.7f);
		require((metalFresnel(f0, tint, 1) - f0).abs().maxCoeff() < 1e-6f,
				"F82 normal incidence color");
		Spectrum edge = (f0 + (Spectrum(1) - f0) * pow5(6.f / 7)) * tint;
		require((metalFresnel(f0, tint, 1.f / 7) - edge).abs().maxCoeff() < 1e-6f,
				"F82 grazing tint");
		for (float roughness : {0.f, .2f, .6f, 1.f})
			for (float mu : {.1f, .5f, 1.f}) {
				Vector3f wo(sqrtf(1 - mu * mu), 0.f, mu);
				double energy	= 0;
				constexpr int n = 160;
				for (int row = 0; row < n; ++row)
					for (int column = 0; column < n; ++column) {
						float z = (row + .5f) / n, phi = 2 * M_PI * (column + .5f) / n;
						Vector3f wi(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
						Spectrum forward = diffuse(Spectrum(1), roughness, wo, wi);
						Spectrum reverse = diffuse(Spectrum(1), roughness, wi, wo);
						require((forward - reverse).abs().maxCoeff() < 1e-5f,
								"EON diffuse reciprocity");
						energy += forward[0] * z * (2 * M_PI / (n * n));
					}
				require(fabs(energy - 1) < .006, "EON white furnace");
			}
		for (float roughness : {.15f, .5f, 1.f}) {
			float alpha = roughness * roughness;
			require(opaqueEnergy(1.f, alpha, .5f) > .999f,
					"Index matched diffuse has no interface loss");
			require(opaqueAverage(1.5f, alpha) > 0 && opaqueAverage(1.5f, alpha) <= 1,
					"Opaque energy LUT range");
			GGX ggx(alpha, .65f);
			Vector3f wo = normalize(Vector3f(.7f, .2f, .6f));
			for (int sample = 0; sample < 2048; ++sample) {
				Vector2f u((sample + .5f) / 2048, fmodf(sample * .61803398875f + .25f, 1.f));
				Vector3f m = ggx.sampleNormal(wo, u);
				require(fabsf(m.norm() - 1) < 1e-5f && m[2] >= 0 && dot(m, wo) >= 0,
						"GGX visible normal sampling");
				Vector3f wi = 2 * dot(wo, m) * m - wo;
				if (wi[2] <= 0) continue;
				float expected = ggx.normalPdf(wo, m) / (4 * dot(wi, m));
				require(ggx.reflectionPdf(wo, wi) ==
					ggx.reflectionPdf(wo, wi, normalize(wo + wi)),
					"Shared GGX reflection density");
				require(fabsf(expected - ggx.reflectionPdf(wo, wi)) < 1e-3f * fmaxf(1, expected),
						"GGX reflection Jacobian");
			}
		}
		Vector3f wo		= normalize(Vector3f(.7f, .2f, .6f));
		Vector3f ltc	= fuzzCoefficients(.7f, wo[2]);
		double integral = 0;
		constexpr int n = 256;
		for (int row = 0; row < n; ++row)
			for (int column = 0; column < n; ++column) {
				float z = (row + .5f) / n, phi = 2 * M_PI * (column + .5f) / n;
				Vector3f wi(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
				integral += fuzzPdf(wi, wo, ltc) * (2 * M_PI / (n * n));
			}
		require(fabs(integral - 1) < .001, "LTC fuzz PDF normalization");
		for (int index = 0; index < 1024; ++index) {
			Vector3f wi = sampleFuzz(
				{(index + .5f) / 1024, fmodf(index * .61803398875f + .25f, 1.f)}, wo, ltc);
			require(fabsf(wi.norm() - 1) < 1e-5f && fuzzPdf(wi, wo, ltc) > 0, "LTC sample support");
		}
		std::cout << "OpenPBR lobe math passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
