#pragma once
#include "common.h"
#include "taggedptr.h"
#include "render/phase.h"
#include "medium.h"

KRR_NAMESPACE_BEGIN

class Ray;

struct MediumProperties {
	Color sigma_a, sigma_s;
	PhaseFunction phase;
	Color Le;
};

struct RayMajorantSegment {
	float tMin, tMax;
	Color sigma_maj;
};

class RayMajorantIterator {
public:
	KRR_CALLABLE RayMajorantIterator(float tMin, float tMax, const Color &sigma_maj) :
		tMin(tMin), tMax(tMax), sigma_maj(sigma_maj) {}

	KRR_CALLABLE RayMajorantSegment next() { return {tMin, tMax, sigma_maj}; }

	float tMin, tMax;
	Color sigma_maj;
};

class HomogeneousMedium {
public:

	KRR_CALLABLE bool isEmissive() const { return !Le.isZero(); }

	KRR_CALLABLE MediumProperties samplePoint(const Vector3f &p) const {
		return { sigma_a, sigma_s, &phase, Le };
	}

	KRR_CALLABLE RayMajorantIterator sampleRay(const Ray &ray, float tMax) {
		return {0, tMax, sigma_a + sigma_s};
	}

	Color sigma_a, sigma_s, Le;
	HGPhaseFunction phase;
};

class MediumInterface {
public:
	KRR_CALLABLE MediumInterface() = default;
	KRR_CALLABLE MediumInterface(Medium m) : inside(m), outside(m) {}
	KRR_CALLABLE MediumInterface(Medium mi, Medium mo) : inside(mi), outside(mo) {}

	KRR_CALLABLE bool isTransition() const { return inside != outside; }

	Medium inside, outside;
};

KRR_NAMESPACE_END