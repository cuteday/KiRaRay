#pragma once

#include "common.h"
#include "taggedptr.h"
#include "shape.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

struct LightSample {
	Interaction intr;
	vec3f L;
	float pdf;
};

struct LightSampleContext {
	vec3f p;
	vec3f n;
};

class DiffuseAreaLight {
public:
	DiffuseAreaLight() = default;

	DiffuseAreaLight(Shape& shape, Texture& texture, vec3f Le = 0.f,
		bool twoSided = true, float scale = 1.f) :
		shape(shape),
		texture(texture),
		Le(Le),
		twoSided(twoSided),
		scale(scale) {}

	__device__ inline LightSample sampleLi(vec2f u, const LightSampleContext& ctx) const {
		LightSample ls = {};
		
		ShapeSampleContext shapeCtx = { ctx.p, ctx.n };
		ShapeSample ss = shape.sample(u, shapeCtx);
		assert(!isnan(ss.pdf));

		Interaction& intr = ss.intr;
		intr.wo = normalize(ctx.p - intr.p);

		ls.intr = intr;
		ls.pdf = ss.pdf;
		ls.L = L(intr.p, intr.n, intr.uv, intr.wo);
		return ls;
	}

	__device__ inline vec3f L(vec3f p, vec3f n, vec2f uv, vec3f w) const {
		if (!twoSided && dot(n, w) < 0.f) return 0;	// hit backface

		if (texture.isValid()) {
			return scale * texture.tex(uv);
		}
		else {
			return scale * Le;
		}
	}

	KRR_CALLABLE float pdfLi(Interaction& p, LightSampleContext& ctx) const {
		ShapeSampleContext shapeCtx = { ctx.p, ctx.n };
		return shape.pdf(p, shapeCtx);
	}

private:
	Shape shape;
	Texture texture{};		// emissive image texture
	vec3f Le{0};
	bool twoSided{true};
	float scale{1};
};

class Light :public TaggedPointer<DiffuseAreaLight> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE LightSample sampleLi(vec2f u, const LightSampleContext& ctx) const {
		auto sampleLi = [&](auto ptr) -> LightSample {return ptr->sampleLi(u, ctx); };
		return dispatch(sampleLi);
	}

	KRR_CALLABLE vec3f L(vec3f p, vec3f n, vec2f uv, vec3f w)const {
		auto L = [&](auto ptr) -> vec3f { return ptr->L(p, n, uv, w); };
		return dispatch(L);
	}
	
	KRR_CALLABLE float pdfLi(Interaction& p, LightSampleContext& ctx) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdfLi(p, ctx); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END