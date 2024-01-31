#include "common.h"

#include "krrmath/math.h"
#include "render/sampling.h"

NAMESPACE_BEGIN(krr)

// von Mise-Fisher distribution in S2
class VMFDistribution {
public:
	VMFDistribution() = default;
	
	KRR_CALLABLE VMFDistribution(float kappa) : m_kappa(kappa) {}

	KRR_CALLABLE void setKappa(float kappa) { m_kappa = kappa; } 

	KRR_CALLABLE float getKappa(float kappa) const { return m_kappa; }

	KRR_CALLABLE float eval(float cosTheta) const { 
		if (m_kappa < M_EPSILON)
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
	KRR_CALLABLE float eval(const Vector3f& wi, float theta, float phi) const {
		return eval(wi, utils::sphericalToCartesian(theta, phi));
	}
	
	// sample around +z
	KRR_CALLABLE Vector3f sample(Vector2f u) const {
		if (m_kappa < M_EPSILON)
			return uniformSampleSphere(u);
		float cosTheta = 1 + log1p(-u[0] + expf(-2 * m_kappa) * u[0]) / m_kappa;
		float sinTheta = safe_sqrt(1 - cosTheta * cosTheta), sinPhi, cosPhi;
		float phi	   = M_2PI * u[1];
		sinPhi		   = sin(phi);
		cosPhi		   = cos(phi);
		return Vector3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	// sample and rotate to specified mu
	// @param mu: cartesian coord of the mean direction
	KRR_CALLABLE Vector3f sample(Vector2f u, Vector3f mu) const { 
		return Frame(mu).toWorld(sample(u));
	}
	
	// sample and rotate to specified mu
	// @param mu: spherical coord [theta, phi] of the mean direction
	KRR_CALLABLE Vector3f sample(Vector2f u, float theta, float phi) const {
		Vector3f dir = sample(u);
		Vector3f mu_cartesian = utils::sphericalToCartesian(theta, phi);
		return Frame(mu_cartesian).toWorld(dir);
	}
	
private:
	float m_kappa{};
};

NAMESPACE_END(krr)