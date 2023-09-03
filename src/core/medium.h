#pragma once
#include "common.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

class Ray;

class HomogeneousMedium;
class MediumProperties;
class RayMajorantIterator;

class Medium : public TaggedPointer<HomogeneousMedium> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE Color Le(Vector3f p) const;

	KRR_CALLABLE bool isEmissive() const;

	KRR_CALLABLE MediumProperties samplePoint(Vector3f p) const;

	KRR_CALLABLE RayMajorantIterator sampleRay(const Ray &ray, float tMax);
};

KRR_NAMESPACE_END