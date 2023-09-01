#pragma once
#include "common.h"
#include "taggedptr.h"
#include "render/phase.h"

KRR_NAMESPACE_BEGIN

namespace rt {

class HomogeneousMedium {
public:

	KRR_CALLABLE bool isEmissive() const {}

	KRR_CALLABLE void samplePoint(const Vector3f &p) const {}

	KRR_CALLABLE void sampleRay(const Ray &ray, float tMax) {}

	Color sigma_a, sigma_s, Le;
	HGPhaseFunction phase;
};

class Medium : public TaggedPointer<HomogeneousMedium> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE bool isEmissive() const {
		auto emissive = [&](auto ptr) -> bool { return ptr->isEmissive(); };
		return dispatch(emissive);
	}

	KRR_CALLABLE void samplePoint(const Vector3f& p) const {
		auto sample = [&](auto ptr) -> void { return ptr->samplePoint(p); };
		return dispatch(sample);
	}

	KRR_CALLABLE void sampleRay(const Ray& ray, float tMax) {
		auto sample = [&](auto ptr) -> void { return ptr->sampleRay(ray, tMax); };
		return dispatch(sample);
	}
};

class MediumInterface {
public:
	KRR_CALLABLE MediumInterface() = default;
	KRR_CALLABLE MediumInterface(Medium m) : inside(m), outside(m) {} 
	KRR_CALLABLE MediumInterface(Medium mi, Medium mo) : inside(mi), outside(mo) {}

	KRR_CALLABLE bool isTransition() const { return inside != outside; }

	Medium inside, outside;
};

} // namespace rt

KRR_NAMESPACE_END