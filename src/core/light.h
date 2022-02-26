#pragma once

#include "common.h"
#include "taggedptr.h"
#include "shape.h"

KRR_NAMESPACE_BEGIN

class DiffuseAreaLight {
public:
	DiffuseAreaLight() = default;

	__both__ inline void sampleLi() {
	}

	__both__ inline vec3f L() {
		return 0;
	}

	__both__ inline float pdf() {
		return 0;
	}
private:
	Shape shape;

};

class Light :TaggedPointer<DiffuseAreaLight> {
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline void sampleLi() {
		auto sampleLi = [&](auto ptr) -> void {return ptr->sampleLi(); };
		return dispatch(sampleLi);
	}

	__both__ inline vec3f L() {
		auto L = [&](auto ptr) -> vec3f { return ptr->L(); };
		return dispatch(L);
	}

	__both__ inline float pdf() {
		auto pdf = [&](auto ptr) -> float { return ptr->pdf(); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END