#pragma once

// Native spectral subset of Adobe OpenPBR BSDF; see NOTICE for intentional differences.
#include "math.h"
#include "render/materials/bxdf.h"
#include "material/program.h"
#include "raytracing.h"

NAMESPACE_BEGIN(krr)
namespace openpbr {

KRR_CALLABLE Vector3f vector(MaterialValue value) { return {value[0], value[1], value[2]}; }
KRR_CALLABLE Vector3f unit(Vector3f value, Vector3f fallback) {
	return value.squaredNorm() > 1e-12f ? normalize(value) : fallback;
}
KRR_CALLABLE Frame frame(Vector3f n, Vector3f tangent, Vector3f bitangent, Vector3f wo) {
	n = unit(n, Vector3f(0.f, 0.f, 1.f));
	if (dot(n, wo) < 0) n = -n;
	Vector3f t = unit(tangent - n * dot(n, tangent), utils::getPerpendicular(n));
	Vector3f b = cross(n, t);
	if (dot(b, bitangent) < 0) b = -b;
	return {n, t, b};
}

// Color algebra is evaluated at the renderer's sampled wavelengths.
class Model {
public:
	KRR_CALLABLE Model() : baseGgx(1), coatGgx(1) {}
	KRR_CALLABLE Model(const MaterialValues &values, const MaterialContext &context, Vector3f wo,
					   const SampledWavelengths &lambda, const RGBColorSpace &colorSpace,
					   MaterialModel model) :
		baseGgx(1), coatGgx(1), view(wo), preview(model == MaterialModel::PreviewSurface) {
		using P		= MaterialParameter;
		auto scalar = [&](P p) { return values[p][0]; };
		auto color	= [&](P p) {
			 MaterialValue v = values[p];
			 return Spectrum::fromRGB(RGB(v[0], v[1], v[2]).cwiseMax(0.f).cwiseMin(1.f),
									  SpectrumType::RGBBounded, lambda, colorSpace);
		};
		thin			 = scalar(P::ThinWalled) != 0;
		inside			 = dot(wo, vector(context.normal)) < 0 && !thin;
		Vector3f tangent = vector(values[P::Tangent]), bitangent = vector(context.bitangent);
		coat		  = saturate(scalar(P::CoatWeight));
		fuzz		  = inside || preview ? 0 : saturate(scalar(P::FuzzWeight));
		baseFrame	  = frame(vector(values[P::Normal]), tangent, bitangent, wo);
		coatFrame	  = coat > 0 ? frame(vector(values[P::CoatNormal]), tangent, bitangent, wo)
								: baseFrame;
		fuzzRoughness = saturate(scalar(P::FuzzRoughness));
		fuzzFrame = baseFrame;
		fuzzData = Vector3f(0.f);
		fuzzColor = Spectrum(0);
		if (fuzz > 0) {
			fuzzFrame = frame(unit((1 - coat) * baseFrame.N + coat * coatFrame.N, baseFrame.N),
				tangent, bitangent, wo);
			fuzzData = fuzzCoefficients(fuzzRoughness, fmaxf(0, dot(wo, fuzzFrame.N)));
			fuzzColor = color(P::FuzzColor);
		}
		baseColor = bounded(color(P::BaseColor) * fmaxf(0, scalar(P::BaseWeight)));
		specularColor	   = color(P::SpecularColor);
		coatColor		   = coat > 0 ? color(P::CoatColor) : Spectrum(1);
		float metalness	   = saturate(scalar(P::Metalness)),
			  transmission = saturate(scalar(P::TransmissionWeight));
		trans			   = (1 - metalness) * transmission;
		transmissionColor  = trans > 0 ? color(P::TransmissionColor) : Spectrum(0);
		opaque			   = (1 - metalness) * (1 - transmission);
		metal			   = metalness * fmaxf(0, scalar(P::SpecularWeight));
		coatEta			   = fmaxf(.01f, scalar(P::CoatIor));
		float eta		   = fmaxf(.01f, scalar(P::SpecularIor));
		float weightedCoat = inside ? 1 : lerp(1.f, coatEta, coat);
		float weightedSpec = preview
								 ? eta
								 : weightedCoat * weightedIor(eta / weightedCoat,
															  fmaxf(0, scalar(P::SpecularWeight)));
		etaRefract		   = inside ? 1 / weightedSpec : weightedSpec;
		etaReflect		   = inside ? 1 / weightedSpec : weightedSpec / weightedCoat;
		if (!inside && weightedCoat > weightedSpec && weightedSpec >= 1)
			etaReflect = weightedCoat / weightedSpec;
		etaOpaque  = inside ? weightedSpec : etaReflect;
		etaRefract = etaRefract >= 1 ? fmaxf(1.0001f, etaRefract) : fminf(1 / 1.0001f, etaRefract);
		diffuseRoughness	= saturate(scalar(P::DiffuseRoughness));
		float coatRoughness = saturate(scalar(P::CoatRoughness));
		float roughness		= saturate(scalar(P::SpecularRoughness));
		if (!preview && coat > 0) {
			if (fuzz > 0) {
				MaterialValue tint = values[P::FuzzColor];
				float fuzzFactor = (tint[0] + tint[1] + tint[2]) / 3 * fuzzRoughness * .005f;
				coatRoughness = lerp(coatRoughness,
					sqrtf(sqrtf(fminf(1, square(square(coatRoughness)) +
						fuzzFactor * square(square(fuzzRoughness))))), fuzz);
			}
			float coatFactor = 1 - (coatEta >= 1 ? 1 / coatEta : coatEta);
			roughness		 = lerp(roughness,
									sqrtf(sqrtf(fminf(1, square(square(roughness)) +
															 coatFactor * square(square(coatRoughness))))),
									coat);
		}
		alpha			  = square(roughness);
		coatAlpha		  = square(coatRoughness);
		baseGgx			  = GGX(alpha, preview ? 0 : scalar(P::SpecularAnisotropy));
		if (coat > 0) coatGgx = GGX(coatAlpha);
		Spectrum metalAvg = metal > 0 ? metalAverage(baseColor, specularColor) : Spectrum(0);
		metalMs			  = metalAvg * metalAvg * metal;
		float cosGeometry = fabsf(dot(wo, vector(context.normal)));
		if (thin && trans > 0) {
			float cosRefract2 = 1 - (1 - square(cosGeometry)) / square(etaRefract);
			transmissionColor = cosRefract2 <= 0
									? Spectrum(0)
									: Spectrum(transmissionColor.pow(1 / sqrtf(cosRefract2)));
			float f			  = fresnel(etaReflect, cosGeometry);
			thinReflect		  = 2 * f / (1 + f);
			f				  = fresnel(etaRefract, cosGeometry);
			thinTransmit	  = 1 - 2 * f / (1 + f);
		}
		Spectrum darkening(1);
		if (!preview && coat > 0) {
			float ks = averageFresnel(coatEta), kr = 1 - (1 - ks) / square(coatEta);
			float fo = averageFresnel(etaOpaque), ft = averageFresnel(etaReflect);
			float baseRoughness = lerp(1.f, roughness, opaque * fo + trans * ft + metalness);
			float k				= lerp(ks, kr, baseRoughness);
			Spectrum eb =
				metal * metalAvg + opaque * (baseColor * (1 - fo) + Spectrum(fo)) + Spectrum(trans);
			darkening =
				(1 - coat) * Spectrum(1) + coat * (1 - k) / (Spectrum(1) - eb * k).cwiseMax(1e-7f);
		}
		float coatMu = fmaxf(0, dot(wo, coatFrame.N));
		float reflected = coat > 0
			? coat * (preview ? fresnel(1.5f, coatMu) : 1 - opaqueEnergy(coatEta, coatAlpha, coatMu))
			: 0;
		coatIncoming	= passage(coatMu) * (1 - reflected) * darkening;
		fuzzAttenuation = 1 - fuzz * fuzzData[2];
		float baseMu	= fmaxf(0, dot(wo, baseFrame.N));
		float loss =
			alpha >= .0016f && !preview && (metal > 0 || (thin && trans > 0))
				? lut2(idealMetal, sqrtf(alpha) * 31, baseMu * 31) : 0;
		float dielectricLoss =
			alpha >= .0016f && !preview && !thin && trans > 0
				? lut3(idealDielectric, iorIndex(etaRefract), sqrtf(alpha) * 31, baseMu * 31)
				: 0;
		weights[0] =
			opaque * baseColor.maxCoeff() + loss * metalMs.maxCoeff() +
			(thin ? loss * trans * thinReflect * specularColor.maxCoeff() : dielectricLoss * trans);
		weights[1] = metal + opaque + trans > 0 ? .05f + reflection(baseMu).maxCoeff() : 0;
		weights[2] = trans > 0 ? .05f + (transmissionColor * trans).maxCoeff() : 0;
		weights[3] = thin ? loss * trans * thinTransmit * transmissionColor.maxCoeff()
						  : dielectricLoss * trans;
		weights[4] = inside ? 0 : coat * (.05f + reflected);
		weights[5] = fuzz * fuzzData[2] * fuzzColor.maxCoeff();
		float sum  = 0;
		for (float weight : weights) sum += weight;
		if (sum > 0)
			for (float &weight : weights) weight /= sum;
	}

	KRR_CALLABLE Spectrum passage(float mu) const {
		if (mu <= 0 || coat == 0 || preview) return Spectrum(1);
		float ct2 = 1 - (1 - square(mu)) / square(coatEta);
		if (ct2 <= 0) return Spectrum(1 - coat);
		return Spectrum(1 - coat) + coat * coatColor.pow(.5f / sqrtf(ct2));
	}
	KRR_CALLABLE Spectrum emissionScale() const {
		return inside ? Spectrum(0) : Spectrum(coatIncoming * fuzzAttenuation);
	}
	KRR_CALLABLE Spectrum reflection(float mu) const {
		if (preview) {
			float f0   = square((etaOpaque - 1) / (etaOpaque + 1));
			Spectrum c = baseColor * metal + Spectrum(f0 * (1 - metal));
			return c + (Spectrum(1) - c) * pow5(1 - mu);
		}
		Spectrum result(0);
		if (metal > 0) result = metal * metalFresnel(baseColor, specularColor, mu);
		if (opaque > 0) result += specularColor * (opaque * fresnel(etaOpaque, mu));
		if (trans > 0)
			result += thin ? specularColor * (trans * thinReflect)
				: (inside ? Spectrum(trans) : specularColor * trans) * fresnel(etaReflect, mu);
		return result;
	}
	KRR_CALLABLE Spectrum baseCos(Vector3f wi, TransportMode mode) const {
		Vector3f wo = baseFrame.toLocal(view);
		wi			= baseFrame.toLocal(wi);
		if (wo[2] <= 0 || wi[2] == 0) return Spectrum(0);
		Spectrum result(0);
		if (wi[2] > 0) {
			float mu = fmaxf(0, dot(wo, normalize(wo + wi)));
			result	 = reflection(mu) * baseGgx.reflectionCos(wo, wi);
			if (opaque > 0) {
				if (preview)
					result += baseColor * (opaque * M_INV_PI * wi[2]);
				else
					result += diffuse(baseColor * opaque, diffuseRoughness, wo, wi) * wi[2] *
							  opaqueEnergy(etaOpaque, alpha, wo[2]) *
							  opaqueEnergy(etaOpaque, alpha, wi[2]) /
							  fmaxf(1e-7f, opaqueAverage(etaOpaque, alpha));
			}
			if (!preview && alpha >= .0016f && (metal > 0 || (thin && trans > 0))) {
				float a		 = sqrtf(alpha) * 31;
				float factor = lut2(idealMetal, a, wo[2] * 31) * lut2(idealMetal, a, wi[2] * 31) /
							   fmaxf(1e-7f, lut1(averageMetal, a));
				Spectrum scale = metalMs;
				if (thin) scale += specularColor * (trans * thinReflect);
				result += scale * (fminf(factor, 1 / wi[2]) * M_INV_PI * wi[2]);
			}
		} else if (trans > 0) {
			if (thin) {
				Vector3f flipped(wi[0], wi[1], -wi[2]);
				result +=
					transmissionColor * (trans * thinTransmit * baseGgx.reflectionCos(wo, flipped));
				if (alpha >= .0016f) {
					// A flipped GGX sheet needs reflection energy compensation, not refraction's
					// LUT.
					float a = sqrtf(alpha) * 31, mu = -wi[2];
					float factor = lut2(idealMetal, a, wo[2] * 31) * lut2(idealMetal, a, mu * 31) /
								   fmaxf(1e-7f, lut1(averageMetal, a));
					result += transmissionColor *
							  (trans * thinTransmit * fminf(factor, 1 / mu) * M_INV_PI * mu);
				}
			} else {
				Vector3f m;
				float jacobian = transmissionJacobian(wo, wi, m);
				if (jacobian > 0) {
					float f		 = 1 - fresnel(etaReflect, fabsf(dot(wo, m)));
					float factor = baseGgx.normalPdf(wo, m) * jacobian * baseGgx.G1(wi);
					if (mode == TransportMode::Radiance) factor /= square(etaRefract);
					result += transmissionColor * (trans * f * factor);
				}
			}
		}
		if (!preview && !thin && trans > 0 && alpha >= .0016f) {
			float a = sqrtf(alpha) * 31, mu = fabsf(wi[2]);
			float eta	 = wi[2] > 0 ? etaRefract : 1 / etaRefract;
			float ratio	 = lut2(reflectionRatio, iorIndex(etaRefract), a);
			float factor = lut3(idealDielectric, iorIndex(etaRefract), a, wo[2] * 31) *
						   lut3(idealDielectric, iorIndex(eta), a, mu * 31) /
						   fmaxf(1e-7f, lut2(averageDielectric, iorIndex(eta), a));
			factor *= wi[2] > 0 ? ratio : 1 - ratio;
			Spectrum tint = wi[2] > 0 ? (inside ? Spectrum(1) : specularColor) : transmissionColor;
			if (wi[2] < 0 && !thin && mode == TransportMode::Radiance) factor /= square(etaRefract);
			result += tint * (trans * fminf(factor, 1 / mu) * M_INV_PI * mu);
		}
		return result;
	}
	KRR_CALLABLE float transmissionJacobian(Vector3f wo, Vector3f wi, Vector3f &m) const {
		Vector3f h = wo + etaRefract * wi;
		if (h.squaredNorm() <= 1e-20f) return 0;
		m = normalize(h);
		if (m[2] < 0) m = -m;
		float a = dot(wo, m), b = dot(wi, m);
		if (a <= 0 || b >= 0) return 0;
		float denominator = a + etaRefract * b;
		if (denominator == 0) return 0;
		return fabsf(square(etaRefract) * b / square(denominator));
	}
	KRR_CALLABLE Spectrum evalCos(Vector3f wi, TransportMode mode) const {
		Spectrum result = baseCos(wi, mode);
		float mu		= dot(wi, coatFrame.N);
		if (inside) return result * passage(-mu);
		result *= coatIncoming * passage(mu);
		if (coat > 0 && mu > 0) {
			Vector3f woCoat = coatFrame.toLocal(view), wiCoat = coatFrame.toLocal(wi);
			float f = fresnel(coatEta, fmaxf(0, dot(woCoat, normalize(woCoat + wiCoat))));
			result += Spectrum(coat * f * coatGgx.reflectionCos(woCoat, wiCoat));
		}
		result *= fuzzAttenuation;
		if (fuzz > 0)
			result +=
				fuzzColor * (fuzz * fuzzData[2] *
							 fuzzPdf(fuzzFrame.toLocal(wi), fuzzFrame.toLocal(view), fuzzData));
		return result;
	}
	KRR_CALLABLE float pdf(Vector3f wi) const {
		Vector3f woBase = baseFrame.toLocal(view), wiBase = baseFrame.toLocal(wi);
		if (woBase[2] <= 0) return 0;
		float result = 0;
		if (wiBase[2] > 0)
			result = weights[0] * wiBase[2] * M_INV_PI +
					 weights[1] * baseGgx.reflectionPdf(woBase, wiBase);
		else if (wiBase[2] < 0) {
			result = weights[3] * -wiBase[2] * M_INV_PI;
			if (thin && weights[2] > 0)
				result +=
					weights[2] * baseGgx.reflectionPdf(woBase, {wiBase[0], wiBase[1], -wiBase[2]});
			else if (weights[2] > 0) {
				Vector3f m;
				float j = transmissionJacobian(woBase, wiBase, m);
				if (j > 0) result += weights[2] * baseGgx.normalPdf(woBase, m) * j;
			}
		}
		if (weights[4] > 0)
			result +=
				weights[4] * coatGgx.reflectionPdf(coatFrame.toLocal(view), coatFrame.toLocal(wi));
		if (weights[5] > 0)
			result +=
				weights[5] * fuzzPdf(fuzzFrame.toLocal(wi), fuzzFrame.toLocal(view), fuzzData);
		return result;
	}
	KRR_CALLABLE Vector3f sample(float choice, Vector2f u) const {
		int lobe = 5;
		for (int index = 0; index < 6; ++index) {
			if (choice < weights[index]) {
				lobe = index;
				break;
			}
			choice -= weights[index];
		}
		if (weights[lobe] == 0) return Vector3f(0.f);
		if (lobe == 5) return fuzzFrame.toWorld(sampleFuzz(u, fuzzFrame.toLocal(view), fuzzData));
		if (lobe == 0 || lobe == 3) {
			Vector3f wi = cosineSampleHemisphere(u);
			if (lobe == 3) wi[2] = -wi[2];
			return baseFrame.toWorld(wi);
		}
		Frame f		= lobe == 4 ? coatFrame : baseFrame;
		GGX ggx		= lobe == 4 ? coatGgx : baseGgx;
		Vector3f wo = f.toLocal(view), m = ggx.sampleNormal(wo, u);
		float mu = dot(wo, m);
		if (mu <= 0) return Vector3f(0.f);
		Vector3f wi = 2 * mu * m - wo;
		if (lobe == 2) {
			if (thin) {
				if (wi[2] <= 0) return Vector3f(0.f);
				wi[2] = -wi[2];
			} else {
				float ct2 = 1 - (1 - square(mu)) / square(etaRefract);
				if (ct2 <= 0) return Vector3f(0.f);
				wi = -wo / etaRefract + (mu / etaRefract - sqrtf(ct2)) * m;
			}
			if (wi[2] >= 0) return Vector3f(0.f);
		} else if (wi[2] <= 0)
			return Vector3f(0.f);
		return f.toWorld(normalize(wi));
	}
	KRR_CALLABLE bool transmissive() const { return trans > 0; }

	Frame baseFrame, coatFrame, fuzzFrame;
	GGX baseGgx, coatGgx;
	Vector3f view, fuzzData;
	Spectrum baseColor, specularColor, coatColor, transmissionColor, fuzzColor, metalMs,
		coatIncoming;
	float opaque{0}, trans{0}, metal{0}, coat{0}, fuzz{0}, alpha{0}, coatAlpha{0};
	float etaRefract{1.5f}, etaReflect{1.5f}, etaOpaque{1.5f}, coatEta{1.5f};
	float diffuseRoughness{0}, fuzzRoughness{0}, thinReflect{0}, thinTransmit{0},
		fuzzAttenuation{1};
	float weights[6]{};
	bool thin{false}, inside{false}, preview{false};
};

} // namespace openpbr
NAMESPACE_END(krr)
