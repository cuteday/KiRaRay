#include "common.h"

#include "math/math.h"
#include "raytracing.h"

#include "render/sampling.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

// von Mise-Fisher distribution in S2
class VMFDistribution {
public:
	VMFDistribution() = default;
	
	KRR_CALLABLE VMFDistribution(float kappa) : m_kappa(kappa) {}

	KRR_CALLABLE void setKappa(float kappa) { m_kappa = kappa; } 

	KRR_CALLABLE float getKappa(float kappa) const { return m_kappa; }

	KRR_CALLABLE float eval(float cosTheta) const { 
		if (m_kappa == 0.f)
			return M_INV_4PI;
		return expf(m_kappa * min(0.f, cosTheta - 1.f)) * m_kappa /
			   (M_2PI * (1 - expf(-2 * m_kappa)));
	}

	// @param wi: normalized direction of the incoming ray, in local space.
	// @returns pdf of the distribution.
	KRR_CALLABLE float eval(const Vector3f& wi) const { 
		return eval(wi[2]);
	}

	// Evaluate the SG with its specified mean direction.
	// @param mu: spherical coord [theta, phi] of the mean direction.
	KRR_CALLABLE float eval(const Vector3f& wi, const Vector3f& mu) const { 
		return eval(dot(wi, mu)); 
	}

	// Evaluate the SG with its specified mean direction.
	// @param mu: cartesian coord of the mean direction.
	KRR_CALLABLE float eval(const Vector3f& wi, const Vector2f& mu) const {
		return eval(wi, utils::sphericalToCartesian(mu[0], mu[1]));
	}
	
	// sample around +z
	KRR_CALLABLE Vector3f sample(Vector2f u) const {
		if (m_kappa < M_EPSILON)
			return uniformSampleSphere(u);
		float cosTheta = clamp(1 + log(u[0] + exp(-2 * m_kappa) * (1 - u[0])) / m_kappa, -1.f, 1.f);
		float sinTheta = safe_sqrt(1 - cosTheta * cosTheta), sinPhi, cosPhi;
		sinPhi		   = sin(M_2PI * u[1]);
		cosPhi		   = cos(M_2PI * u[1]);
		return Vector3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	// sample and rotate to specified mu
	// @param mu: cartesian coord of the mean direction
	KRR_CALLABLE Vector3f sample(Vector2f u, Vector3f mu) { 
		return Frame(mu).toWorld(sample(u));
	}
	
	// sample and rotate to specified mu
	// @param mu: spherical coord [theta, phi] of the mean direction
	KRR_CALLABLE Vector3f sample(Vector2f u, Vector2f mu) {
		Vector3f dir = sample(u);
		Vector3f mu_cartesian = utils::sphericalToCartesian(mu[0], mu[1]);
		return Frame(mu_cartesian).toWorld(dir);
	}
	
private:
	float m_kappa{};
};

KRR_NAMESPACE_END