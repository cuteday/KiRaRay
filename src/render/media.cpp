#include "media.h"

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include "raytracing.h"
#include "medium.h"
#include "util/math_utils.h"
#include "render/phase.h"

KRR_NAMESPACE_BEGIN

KRR_CALLABLE float HGPhaseFunction::evalHenveyGreenstein(float cosTheta, float g) {
	g			= clamp(g, -.99f, .99f);
	float denom = 1 + pow2(g) + 2 * g * cosTheta;
	return M_INV_4PI * (1 - pow2(g)) / (denom * safe_sqrt(denom));
}

KRR_CALLABLE Vector3f HGPhaseFunction::sampleHenveyGreenstein(const Vector3f &wo, float g,
																	 const Vector2f &u,
																	 float *pdf) {
	g = clamp(g, -.99f, .99f);

	// Compute $\cos\theta$ for Henyey--Greenstein sample
	float cosTheta;
	if (fabs(g) < 1e-3f)
		cosTheta = 1 - 2 * u[0];
	else
		cosTheta = -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * u[0])));

	// Compute direction _wi_ for Henyey--Greenstein sample
	float sinTheta = safe_sqrt(1 - pow2(cosTheta));
	float phi	   = M_2PI * u[1];
	Frame wFrame(wo);
	Vector3f wi = wFrame.toWorld(utils::sphericalToCartesian(sinTheta, cosTheta, phi));

	if (pdf) *pdf = evalHenveyGreenstein(cosTheta, g);
	return wi;
}

KRR_CALLABLE bool Medium::isEmissive() const {
	auto emissive = [&](auto ptr) -> bool { return ptr->isEmissive(); };
	return dispatch(emissive);
}

KRR_CALLABLE MediumProperties Medium::samplePoint(const Vector3f &p) const {
	auto sample = [&](auto ptr) -> MediumProperties { return ptr->samplePoint(p); };
	return dispatch(sample);
}

KRR_CALLABLE RayMajorantIterator Medium::sampleRay(const Ray &ray, float tMax) {
	auto sample = [&](auto ptr) -> RayMajorantIterator { return ptr->sampleRay(ray, tMax); };
	return dispatch(sample);
}

KRR_NAMESPACE_END