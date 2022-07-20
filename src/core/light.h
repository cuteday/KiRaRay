#pragma once

#include "common.h"
#include "taggedptr.h"
#include "shape.h"
#include "texture.h"
#include "window.h"


KRR_NAMESPACE_BEGIN

struct LightSample {
	Interaction intr;
	Color L;
	float pdf;
};

struct LightSampleContext {
	Vec3f p;
	Vec3f n;
};

class DiffuseAreaLight {
public:

	DiffuseAreaLight() = default;

	DiffuseAreaLight(Shape &shape, Texture &texture, Vec3f Le = {},
		bool twoSided = true, float scale = 1.f) :
		shape(shape),
		texture(texture),
		Le(Le),
		twoSided(twoSided),
		scale(scale) {}

	__device__ inline LightSample sampleLi(Vec2f u, const LightSampleContext& ctx) const {
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

	__device__ inline Color L(Vec3f p, Vec3f n, Vec2f uv, Vec3f w) const {
		if (!twoSided && dot(n, w) < 0.f) return Vec3f::Zero();	// hit backface

		if (texture.isValid()) {
			return scale * texture.tex(uv);
		}
		else {
			return scale * Le;
		}
	}

	KRR_CALLABLE float pdfLi(Interaction& p, const LightSampleContext& ctx) const {
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

	InfiniteLight(Color tint = Color::Ones(), float scale = 1, float rotation = 0)
		:tint(tint), scale(scale), rotation(rotation) {}

	InfiniteLight(const Texture &image, Vec3f tint = Vec3f::Ones(), float scale = 1, float rotation = 0)
		:image(image), tint(tint), scale(scale), rotation(rotation) {}

	InfiniteLight(const string image, Vec3f tint = Vec3f::Ones(), float scale = 1, float rotation = 0)
		:tint(tint), scale(scale), rotation(rotation) {
		setImage(image);
	}

	__device__ inline LightSample sampleLi(Vec2f u, const LightSampleContext& ctx) const {
		LightSample ls = {};
		Vec3f wi = utils::latlongToWorld(u);
		ls.intr = Interaction(ctx.p + wi * 1e7f);
		ls.L = Li(wi);
		ls.pdf = 0.25 * M_INV_PI;
		return ls;
	}

	KRR_CALLABLE float pdfLi(Interaction& p, const LightSampleContext& ctx) const {
		return 0.25 * M_INV_PI;
	}

	KRR_CALLABLE Color L(Vec3f p, Vec3f n, Vec2f uv, Vec3f w) const { return Vec3f::Zero(); }

	__device__ inline Color Li(Vec3f wi) const {
		Color L;
		L = tint * scale;

		if (!image.isOnDevice()) return L;
		Vec2f uv = utils::worldToLatLong(wi);
		uv[0] = fmod(uv[0] + rotation, 1.f);
		L *= image.tex(uv);

		return L;
	}

	void setImage(const string& filename) {
		logDebug("Loading environment texture from: " + filename);
		image.loadImage(filename);
		image.toDevice();
	}

	void renderUI() {
		ui::SliderFloat("Intensity", &scale, 0, 10, "%.02f");
		ui::SliderFloat("Rotation", &rotation, 0, 1, "%.03f");
		ui::ColorEdit3("Tint", (float*)&tint);
		if(image.isValid()) image.renderUI();
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

	KRR_CALLABLE LightSample sampleLi(Vec2f u, const LightSampleContext& ctx) const {
		auto sampleLi = [&](auto ptr) -> LightSample {return ptr->sampleLi(u, ctx); };
		return dispatch(sampleLi);
	}

	KRR_CALLABLE Color L(Vec3f p, Vec3f n, Vec2f uv, Vec3f w) const {
		auto L = [&](auto ptr) -> Vec3f { return ptr->L(p, n, uv, w); };
		return dispatch(L);
	}
	
	KRR_CALLABLE float pdfLi(Interaction& p, const LightSampleContext& ctx) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdfLi(p, ctx); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END