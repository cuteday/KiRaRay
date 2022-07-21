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
			return M_INV_PI * 0.25;
		return exp(m_kappa * min(0.f, cosTheta - 1.f)) * m_kappa /
			   (2 * M_PI * (1 - exp(-2 * m_kappa)));
	}
	
	// sample around +z
	KRR_CALLABLE Vector3f sample(vec2f u) const {
		if (m_kappa == 0.f)
			return uniformSampleSphere(u);
		float cosTheta =
			1 + (log(u.x + exp(-2 * m_kappa) * (1 - u.x))) / m_kappa;

		float sinTheta = safe_sqrt(1 - cosTheta * cosTheta), sinPhi, cosPhi;
		sinPhi = sin(2 * M_PI * u.y);
		cosPhi = cos(2 * M_PI * u.y);
		return Vector3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	// sample and rotate to specified mu
	KRR_CALLABLE Vector3f sample(vec2f u, Vector3f mu) { 
		Vector3f dir = sample(u);
		return Frame(mu).toWorld(dir);
	}
	
private:
	float m_kappa{};
};

KRR_NAMESPACE_END