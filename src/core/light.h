#pragma once

#include "common.h"
#include "taggedptr.h"
#include "shape.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

struct LightSample {
	Interaction intr;
	Color L;
	float pdf;
};

struct LightSampleContext {
	Vector3f p;
	Vector3f n;
};

class DiffuseAreaLight {
public:

	DiffuseAreaLight() = default;

	DiffuseAreaLight(Shape &shape, Vector3f Le, 
		bool twoSided = true, float scale = 1.f)
		: shape(shape), Le(Le), twoSided(twoSided), scale(scale) {}

	DiffuseAreaLight(Shape &shape, Texture &texture, Vector3f Le = {},
		bool twoSided = true, float scale = 1.f) :
		shape(shape),
		texture(texture),
		Le(Le),
		twoSided(twoSided),
		scale(scale) {}

	__device__ inline LightSample sampleLi(Vector2f u, const LightSampleContext& ctx) const {
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

	__device__ inline Color L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w) const {
		if (!twoSided && dot(n, w) < 0.f) return Color::Zero(); // hit backface

		if (texture.isOnDevice()) {
			return scale * texture.tex(uv);
		}
		else {
			return scale * Le;
		}
	}

	KRR_CALLABLE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const {
		ShapeSampleContext shapeCtx = { ctx.p, ctx.n };
		return shape.pdf(p, shapeCtx);
	}

private:
	Shape shape;
	Texture texture{};		// emissive image texture
	Color Le{ 0 };
	bool twoSided{true};
	float scale{1};
};

class InfiniteLight {
public:

	InfiniteLight() = default;

	InfiniteLight(Color tint, float scale = 1, float rotation = 0)
		:tint(tint), scale(scale), rotation(rotation) {}

	InfiniteLight(const Texture &image, Vector3f tint = Vector3f::Ones(), float scale = 1, float rotation = 0)
		:image(image), tint(tint), scale(scale), rotation(rotation) {}

	InfiniteLight(const string image, Vector3f tint = Vector3f::Ones(), float scale = 1, float rotation = 0)
		:tint(tint), scale(scale), rotation(rotation) {
		setImage(image);
	}

	__device__ inline LightSample sampleLi(Vector2f u, const LightSampleContext& ctx) const {
		LightSample ls = {};
		Vector3f wi	   = uniformSampleSphere(u);
		ls.intr		   = Interaction(ctx.p + wi * 1e7f);
		ls.L		   = Li(wi);
		ls.pdf		   = M_INV_4PI;
		return ls;
	}

	KRR_CALLABLE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const {
		return M_INV_4PI;
	}

	KRR_CALLABLE Color L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w) const {
		return Color::Zero();
	}

	__device__ inline Color Li(Vector3f wi) const {
		Color L = tint * scale;

		if (!image.isOnDevice()) return L;
		Vector2f uv = utils::cartesianToSphericalNormalized(wi);
		uv[0] = fmod(uv[0] + rotation, 1.f);
		L *= image.tex(uv);

		return L;
	}

	void setImage(const string& filename) {
		logDebug("Loading environment texture from: " + filename);
		image.loadImage(filename);
		image.toDevice();
	}

	void renderUI();

	friend void from_json(const json& j, InfiniteLight& p) {
		p.scale	   = j.value("scale", 1.f);
		p.tint	   = j.value("tint", Color{ 1 });
		p.rotation = j.value("rotation", 0);
		if (j.contains("image"))
			p.setImage(j.at("image"));
	}

	friend void to_json(json &j, const InfiniteLight &p) {
		j = json{
			{"scale", p.scale},
			{"tint", p.tint},
			{"rotation", p.rotation}
		};
	}

private: 
	Texture image{};
	float scale{1};
	float rotation{0};
	Color tint{1};
};

class Light :public TaggedPointer<DiffuseAreaLight, InfiniteLight> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE LightSample sampleLi(Vector2f u, const LightSampleContext& ctx) const {
		auto sampleLi = [&](auto ptr) -> LightSample {return ptr->sampleLi(u, ctx); };
		return dispatch(sampleLi);
	}

	KRR_CALLABLE Color L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w) const {
		auto L = [&](auto ptr) -> Vector3f { return ptr->L(p, n, uv, w); };
		return dispatch(L);
	}
	
	KRR_CALLABLE float pdfLi(const Interaction& p, const LightSampleContext& ctx) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdfLi(p, ctx); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END