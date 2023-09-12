#pragma once
#include "common.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

class Ray;

class HGPhaseFunction;
class PhaseFunctionSample;
class HomogeneousMedium;
class NanoVDBMedium;
class MediumProperties;
class RayMajorant;

class PhaseFunction : public TaggedPointer<HGPhaseFunction> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_HOST_DEVICE PhaseFunctionSample sample(const Vector3f &wo, const Vector2f &u) const;

	KRR_HOST_DEVICE float pdf(const Vector3f &wo, const Vector3f &wi) const;

	KRR_HOST_DEVICE float p(const Vector3f &wo, const Vector3f &wi) const;
};


class Medium : public TaggedPointer<HomogeneousMedium, NanoVDBMedium> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_HOST_DEVICE Color Le(Vector3f p) const;

	KRR_HOST_DEVICE bool isEmissive() const;

	KRR_HOST_DEVICE MediumProperties samplePoint(Vector3f p) const;

	KRR_HOST_DEVICE RayMajorant sampleRay(const Ray &ray, float tMax) const;
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