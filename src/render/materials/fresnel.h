#pragma once

#include "common.h"
#include "math/math.h"
#include "matutils.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

namespace bsdf {
	using namespace math;

	KRR_CALLABLE vec3f FrSchlick(vec3f f0, vec3f f90, float cosTheta){
		//return lerp(f0, f90, pow(max(1 - cosTheta, 0.f), 5.f));
		return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0.f), 5.f); // clamp to avoid NaN if cosTheta = 1+epsilon
	}

	KRR_CALLABLE float FrSchlick(float f0, float f90, float cosTheta){
		return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0.f), 5.f); // clamp to avoid NaN if cosTheta = 1+epsilon
	}

	// eta: just etaT/etaI if incident ray
	KRR_CALLABLE float FrDielectric(float cosTheta_i, float eta) {
		// computes the reflected fraction of light, i.e. the Fresnel reflectance
		// while the transmitted energy is 1-Fr due to energy conservation
		cosTheta_i = clamp(cosTheta_i, -1.f, 1.f);
		// Potentially flip interface orientation for Fresnel equations
		if (cosTheta_i < 0) {
			eta = 1 / eta;
			cosTheta_i = -cosTheta_i;
		}
		// Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
		float sin2Theta_i = 1 - pow2(cosTheta_i);
		float sin2Theta_t = sin2Theta_i / pow2(eta);
		if (sin2Theta_t >= 1)       // NONE of the light pass into another medium (>critical angle)
			return 1.f;
		
		float cosTheta_t = sqrt(max(1 - sin2Theta_t, 0.f));

		float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
		float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
		return (pow2(r_parl) + pow2(r_perp)) / 2;
	}

	KRR_CALLABLE float FrComplex(float cosTheta_i, complex<float> eta) {
		// Fresnel for conductors: some of the energy is absorbed by the material and turned into heat.
		using Complex = complex<float>;
		cosTheta_i = clamp(cosTheta_i, 0.f, 1.f);
		// Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
		float sin2Theta_i = 1 - pow2(cosTheta_i);
		Complex sin2Theta_t = sin2Theta_i / pow2(eta);
		Complex cosTheta_t = sqrt(1 - sin2Theta_t);

		Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
		Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
		return (r_parl.norm() + r_perp.norm()) / 2;
	}

	KRR_CALLABLE vec3f FrComplex(float cosTheta_i, vec3f eta, vec3f k) {
		vec3f result;
		for (int i = 0; i < 3; ++i)
			result[i] = FrComplex(cosTheta_i, complex<float>(eta[i], k[i]));
		return result;
	}

	// eta: etaI/etaT if incident ray
	KRR_CALLABLE vec3f DisneyFresnel(const vec3f& R0, float metallic, float eta, float cosI) {
		return lerp(vec3f(FrDielectric(cosI, eta)), FrSchlick(R0, vec3f(1), cosI), metallic);
	}

#if 0
	class FresnelDisney {
		
	};

	class FresnelDielectric {

		float etaI, etaT;
	};

	class FresnelConductor {

		vec3f etaI, etaT, k;
	};

	class FresnelNull {

	};

	class Fresnel : public TaggedPointer<FresnelNull, FresnelDisney, FresnelDielectric> {
	public:
		using TaggedPointer::TaggedPointer;

		vec3f eval(float cosThetaI) const {
			auto eval = [&](auto ptr)->float {return ptr->eval(cosThetaI); };
			return dispatch(eval);
		}
	};
#endif
}
KRR_NAMESPACE_END