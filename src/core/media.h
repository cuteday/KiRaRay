#pragma once
#include "common.h"
#include "taggedptr.h"
#include "render/phase.h"

KRR_NAMESPACE_BEGIN

namespace rt {

class HomogeneousMedium {
public:

	Color sigma_a, sigma_s, Le;
	HGPhaseFunction phase;
};

class Medium : public TaggedPointer<HomogeneousMedium> {
public:

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