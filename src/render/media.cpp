#include "media.h"

#include "medium.h"
#include "raytracing.h"
#include "util/math_utils.h"

KRR_NAMESPACE_BEGIN
/* Note that the function qualifier (e.g. inline) should be consistent between declaration and definition. */

KRR_HOST_DEVICE PhaseFunctionSample HGPhaseFunction::sample(const Vector3f &wo,
														 const Vector2f &u) const {
	float g = clamp(this->g, -.99f, .99f);

	// Compute $\cos\theta$ for Henyey-Greenstein sample
	float cosTheta;
	if (fabs(g) < 1e-3f)
		cosTheta = 1 - 2 * u[0];
	else
		cosTheta = -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * u[0])));

	// Compute direction _wi_ for Henyey-Greenstein sample
	float sinTheta = safe_sqrt(1 - pow2(cosTheta));
	float phi	   = M_2PI * u[1];
	Frame wFrame(wo);
	Vector3f wi = wFrame.toWorld(utils::sphericalToCartesian(sinTheta, cosTheta, phi));
	
	float pdf = this->pdf(wo, wi);
	return PhaseFunctionSample{wi, pdf, pdf};
}

KRR_HOST_DEVICE float HGPhaseFunction::pdf(const Vector3f &wo, const Vector3f &wi) const {
	return p(wo, wi);
}

KRR_HOST_DEVICE float HGPhaseFunction::p(const Vector3f &wo, const Vector3f &wi) const {
	float g			= clamp(this->g, -.99f, .99f);
	float denom = 1 + pow2(g) + 2 * g * wo.dot(wi);
	return M_INV_4PI * (1 - pow2(g)) / (denom * safe_sqrt(denom));
}

KRR_HOST_DEVICE PhaseFunctionSample PhaseFunction::sample(const Vector3f &wo, const Vector2f &u) const {
	auto sample = [&](auto ptr) -> PhaseFunctionSample { return ptr->sample(wo, u); };
	return dispatch(sample);
}

KRR_HOST_DEVICE float PhaseFunction::pdf(const Vector3f &wo, const Vector3f &wi) const {
	auto pdf = [&](auto ptr) -> float { return ptr->pdf(wo, wi); };
	return dispatch(pdf);
}

KRR_HOST_DEVICE float PhaseFunction::p(const Vector3f &wo, const Vector3f &wi) const {
	auto p = [&](auto ptr) -> float { return ptr->p(wo, wi); };
	return dispatch(p);
}

KRR_HOST_DEVICE bool Medium::isEmissive() const {
	auto emissive = [&](auto ptr) -> bool { return ptr->isEmissive(); };
	return dispatch(emissive);
}

KRR_HOST_DEVICE MediumProperties Medium::samplePoint(Vector3f p) const {
	auto sample = [&](auto ptr) -> MediumProperties { return ptr->samplePoint(p); };
	return dispatch(sample);
}

KRR_HOST_DEVICE RayMajorant Medium::sampleRay(const Ray &ray, float tMax) {
	auto sample = [&](auto ptr) -> RayMajorant { return ptr->sampleRay(ray, tMax); };
	return dispatch(sample);
}

KRR_HOST_DEVICE RayMajorant NanoVDBMedium::sampleRay(const Ray &ray, float raytMax) {
	// [TODO] currently we use a coarse majorant for the whole volume
	// but it seems that nanovdb has a built-in hierachical DDA on gpu?
	float tMin, tMax;
	Ray r = inverseTransform * ray;
	if (!bounds.intersect(r.origin, r.dir, raytMax, &tMin, &tMax)) return {};
	return {sigma_maj};
}

KRR_NAMESPACE_END