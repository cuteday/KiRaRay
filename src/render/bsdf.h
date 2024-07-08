#pragma once

#include "common.h"
#include "device/taggedptr.h"
#include "device/variant.h"

#include "materials/null.h"
#include "materials/diffuse.h"
#include "materials/disney.h"
#include "materials/conductor.h"
#include "materials/dielectric.h"

NAMESPACE_BEGIN(krr)

/* Another way to implement gpu polymorphism that is rather tricky, 
*  avoiding initializing a BSDF multiple times within in a scope, 
*  but needs a maximum size of variant to be known at compile time.
*/
class BSDF : public VariantClass<NullBsdf, DiffuseBrdf, DielectricBsdf, ConductorBsdf, DisneyBsdf> {
public:
	using VariantClass::VariantClass;

	KRR_CALLABLE BSDF(const SurfaceInteraction &intr) { setup(intr); }

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) {
		defaultConstruct(static_cast<size_t>(intr.sd.bsdfType));
		auto setup = [&](auto ptr)->void {return ptr->setup(intr); };
		return dispatch(setup);
	}

	// [NOTE] f the cosine theta term in render equation is not contained in f().
	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
						 TransportMode mode = TransportMode::Radiance) const {
		auto f = [&](auto ptr) -> Spectrum { return ptr->f(wo, wi, mode); };
		return dispatch(f);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sample(wo, sg, mode); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		auto pdf = [&](auto ptr)->float {return ptr->pdf(wo, wi, mode); };
		return dispatch(pdf);
	}

	KRR_CALLABLE BSDFType flags() const {
		auto flags = [&](auto ptr) -> BSDFType { return ptr->flags(); };
		return dispatch(flags);
	}
};

/* The following legacy code will possibly be removed in the future. 
*  But it does not bother to take the maximum size among all types.
*/

class BxDF :public TaggedPointer<NullBsdf, DiffuseBrdf, 
	DielectricBsdf, ConductorBsdf, DisneyBsdf>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE static BSDFSample sample(const SurfaceInteraction &intr, Vector3f wo, Sampler &sg,
										  TransportMode mode = TransportMode::Radiance) {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sampleInternal(intr, wo, sg, mode); };
		return dispatch(sample, static_cast<int>(intr.sd.bsdfType));
	}

	KRR_CALLABLE static Spectrum f(const SurfaceInteraction &intr, Vector3f wo, Vector3f wi,
								TransportMode mode = TransportMode::Radiance) {
		auto f = [&](auto ptr) -> Spectrum { return ptr->fInternal(intr, wo, wi, mode); };
		return dispatch(f, static_cast<int>(intr.sd.bsdfType));
	}

	KRR_CALLABLE static float pdf(const SurfaceInteraction &intr, Vector3f wo, Vector3f wi,
								  TransportMode mode = TransportMode::Radiance) {
		auto pdf = [&](auto ptr)->float {return ptr->pdfInternal(intr, wo, wi, mode); };
		return dispatch(pdf, static_cast<int>(intr.sd.bsdfType));
	}

	KRR_CALLABLE static BSDFType flags(const SurfaceInteraction& intr) {
		auto flags = [&](auto ptr)->BSDFType {return ptr->flagsInternal(intr); };
		return dispatch(flags, static_cast<int>(intr.sd.bsdfType));
	}

	KRR_CALLABLE void setup(const SurfaceInteraction &intr) {
		auto setup = [&](auto ptr)->void {return ptr->setup(intr); };
		return dispatch(setup);
	}

	// [NOTE] f the cosine theta term in render equation is not contained in f().
	KRR_CALLABLE Spectrum f(Vector3f wo, Vector3f wi,
						 TransportMode mode = TransportMode::Radiance) const {
		auto f = [&](auto ptr) -> Spectrum { return ptr->f(wo, wi, mode); };
		return dispatch(f);
	}

	KRR_CALLABLE BSDFSample sample(Vector3f wo, Sampler &sg,
								   TransportMode mode = TransportMode::Radiance) const {
		auto sample = [&](auto ptr)->BSDFSample {return ptr->sample(wo, sg, mode); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(Vector3f wo, Vector3f wi,
						   TransportMode mode = TransportMode::Radiance) const {
		auto pdf = [&](auto ptr)->float {return ptr->pdf(wo, wi, mode); };
		return dispatch(pdf);
	}

	KRR_CALLABLE BSDFType flags() const {
		auto flags = [&](auto ptr) -> BSDFType { return ptr->flags(); };
		return dispatch(flags);
	}
};

NAMESPACE_END(krr)