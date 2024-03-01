#pragma once

#include "common.h"
#include "shape.h"
#include "texture.h"
#include "render/spectrum.h"
#include "device/taggedptr.h"

NAMESPACE_BEGIN(krr)
namespace rt {

struct LightSample {
	Interaction intr;
	Spectrum L;
	float pdf;
};

struct LightSampleContext {
	Vector3f p;
	Vector3f n;
};

enum class LightType {
	DeltaPosition,
	DeltaDirection,
	Area,
	Infinite,
};

class PointLight {
public:
	PointLight() = default;

	PointLight(const Vector3f &translation, const RGB &I, float scale = 1, 
		const RGBColorSpace* colorSpace = KRR_DEFAULT_COLORSPACE) :
		position(translation), I(I), scale(scale), colorSpace(colorSpace) {}

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
									const SampledWavelengths &lambda) const {
		Vector3f wi = (position - ctx.p).normalized();
		Spectrum Li = Spectrum::fromRGB(I, SpectrumType::RGBIlluminant, lambda, *colorSpace);
		Li *= scale / (position - ctx.p).squaredNorm();
		return LightSample{Interaction{position}, Li, 1};
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
								 const SampledWavelengths &lambda) const {
		return Spectrum::Zero();
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const { return 0; }

	KRR_DEVICE LightType type() const { return LightType::DeltaPosition; }

	KRR_DEVICE bool isDeltaLight() const { return true; }

private:
	RGB I;
	float scale;
	Vector3f position;
	const RGBColorSpace *colorSpace;
};

class DirectionalLight {
public:
	DirectionalLight() = default;

	DirectionalLight(const Matrix3f &rotation, const RGB &I, float scale = 1,
					 float sceneRadius				 = 1e5,
					 const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		rotation(rotation), I(I), scale(scale), sceneRadius(sceneRadius), colorSpace(colorSpace) {}

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
									const SampledWavelengths &lambda) const {
		/* [NOTE] For shadow rays, if the ray direction is too large, optix trace will have precision problems! 
		(e.g. the ray will self-intersect on the original surface, even if the ray origin has offset) */
		Vector3f wi = rotation * Vector3f{0, 0, 1};
		Vector3f p	= ctx.p + wi * 2 * sceneRadius;
		Spectrum Li =
			scale * Spectrum::fromRGB(I, SpectrumType::RGBIlluminant, lambda, *colorSpace);
		return LightSample{Interaction{p}, Li, 1};
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
					   const SampledWavelengths &lambda) const {
		return Spectrum::Zero();
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const { return 0; }

	KRR_DEVICE LightType type() const { return LightType::DeltaDirection; }

	KRR_DEVICE bool isDeltaLight() const { return true; }

private:
	RGB I;
	float scale;
	float sceneRadius{1e5};
	Matrix3f rotation;
	const RGBColorSpace *colorSpace;
};

class DiffuseAreaLight {
public:
	DiffuseAreaLight() = default;

	DiffuseAreaLight(Shape &shape, Vector3f Le, bool twoSided = false, float scale = 1.f,
					 const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		shape(shape), Le(Le), twoSided(twoSided), scale(scale), colorSpace(colorSpace) {}

	DiffuseAreaLight(Shape &shape, const rt::TextureData &texture, RGB Le = {},
					 bool twoSided = false, float scale = 1.f,
					 const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		shape(shape), texture(texture), Le(Le), twoSided(twoSided), scale(scale), colorSpace(colorSpace) {}

	KRR_DEVICE LightSample DiffuseAreaLight::sampleLi(Vector2f u, const LightSampleContext &ctx,
													  const SampledWavelengths &lambda) const {
		LightSample ls				= {};
		ShapeSampleContext shapeCtx = {ctx.p, ctx.n};
		ShapeSample ss				= shape.sample(u, shapeCtx);
		DCHECK(!isnan(ss.pdf));
		Interaction &intr = ss.intr;
		intr.wo			  = normalize(ctx.p - intr.p);

		ls.intr = intr;
		ls.pdf	= ss.pdf;
		ls.L	= L(intr.p, intr.n, intr.uv, intr.wo, lambda);
		return ls;
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
							  const SampledWavelengths &lambda) const {
		if (!twoSided && dot(n, w) < 0.f) return Spectrum::Zero(); // hit backface

		RGB L = texture.isValid() ? texture.evaluate(uv).head<3>() : Le;
		return scale * Spectrum::fromRGB(L, SpectrumType::RGBIlluminant, lambda, *colorSpace);
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const {
		ShapeSampleContext shapeCtx = {ctx.p, ctx.n};
		return shape.pdf(p, shapeCtx);
	}

	KRR_DEVICE LightType type() const { return LightType::Area; }

	KRR_DEVICE bool isDeltaLight() const { return false; }

private:
	Shape shape;
	rt::TextureData texture{}; // emissive image texture
	RGB Le{0};
	bool twoSided{true};
	float scale{1};
	const RGBColorSpace *colorSpace;
};

class InfiniteLight {
public:
	InfiniteLight() = default;

	InfiniteLight(const Matrix3f &rotation, RGB tint, float scale = 1, float sceneRadius = 1e5f,
				  const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		tint(tint), scale(scale), rotation(rotation), sceneRadius(sceneRadius), colorSpace(colorSpace) {}

	InfiniteLight(const Matrix3f &rotation, const rt::TextureData &image, float scale = 1,
				  float sceneRadius = 1e5f, const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		image(image), tint(RGB::Ones()), scale(scale), rotation(rotation), sceneRadius(sceneRadius), colorSpace(colorSpace) {}

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
										   const SampledWavelengths &lambda) const {
		// [TODO] use intensity importance sampling here.
		LightSample ls = {};
		Vector3f wi	   = uniformSampleSphere(u);
		ls.intr		   = Interaction(ctx.p + wi * 2 * sceneRadius);
		ls.L		   = Li(wi, lambda);
		ls.pdf		   = M_INV_4PI;
		return ls;
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const {
		return M_INV_4PI;
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
					   const SampledWavelengths &lambda) const {
		return Spectrum::Zero();
	}

	KRR_DEVICE Spectrum Li(Vector3f wi, const SampledWavelengths &lambda) const {
		Vector2f uv = worldToLatLong(rotation.transpose() * wi);
		RGB L		= image.isValid() ? tint * image.evaluate(uv).head<3>() : tint;
		return scale * Spectrum::fromRGB(L, SpectrumType::RGBIlluminant, lambda, *colorSpace);
	}

	KRR_DEVICE LightType type() const { return LightType::Infinite; }

	KRR_DEVICE bool isDeltaLight() const { return false; }

private:
	RGB tint{1};
	float scale{1};
	float sceneRadius{1e5};
	Matrix3f rotation;
	rt::TextureData image{};
	const RGBColorSpace *colorSpace;
};

class Light :
	public TaggedPointer<rt::PointLight, rt::DirectionalLight, rt::DiffuseAreaLight,
						 rt::InfiniteLight> {
public:
	using TaggedPointer::TaggedPointer;

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx, 
									const SampledWavelengths& lambda) const {
		auto sampleLi = [&](auto ptr) -> LightSample { return ptr->sampleLi(u, ctx, lambda); };
		return dispatch(sampleLi);
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
					   const SampledWavelengths &lambda) const {
		auto L = [&](auto ptr) -> Spectrum { return ptr->L(p, n, uv, w, lambda); };
		return dispatch(L);
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const {
		auto pdf = [&](auto ptr) -> float { return ptr->pdfLi(p, ctx); };
		return dispatch(pdf);
	}

	KRR_DEVICE LightType type() const { 
		auto type = [&](auto ptr) -> LightType { return ptr->type(); };
		return dispatch(type); 
	}

	KRR_DEVICE bool isDeltaLight() const {
		auto delta = [&](auto ptr) -> bool { return ptr->isDeltaLight(); };
		return dispatch(delta);
	}
};

/* You only have OneShot. */
} // namespace rt
NAMESPACE_END(krr)