#pragma once

#include "common.h"
#include "shape.h"
#include "texture.h"
#include "render/spectrum.h"
#include "render/materials/openpbr/model.h"
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

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const;

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

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const;

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
									const SampledWavelengths &lambda) const {
		/* [NOTE] For shadow rays, if the ray direction is too large, optix trace will have precision problems! 
		(e.g. the ray will self-intersect on the original surface, even if the ray origin has offset) */
		Vector3f wi = rotation * Vector3f::UnitZ();
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

class SpotLight {
public:
	SpotLight() = default;

	SpotLight(const Transformation &transform, const RGB &I, float scale, float innerCone,
			  float outerCone, const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		transform(transform), Iemit(I), scale(scale), cosInnerCone(std::cos(radians(innerCone))),
		cosOuterCone(std::cos(radians(outerCone))), colorSpace(colorSpace) {}

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const;

	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
									const SampledWavelengths &lambda) const {
		Point3f p		   = transform.translation();		
		Vector3f wLight	   = normalize(transform.inverse() * ctx.p);
		//Vector3f wLight  = (transform.inverse().linear() * -(p - ctx.p)).normalized();
		Spectrum Li		   = I(wLight, lambda) / (p - ctx.p).squaredNorm();

		return LightSample{Interaction{p}, Li, 1};
	}

	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
						  const SampledWavelengths &lambda) const {
		return Spectrum::Zero();
	}

	KRR_DEVICE float pdfLi(const Interaction &p, const LightSampleContext &ctx) const { return 0; }

	KRR_DEVICE LightType type() const { return LightType::DeltaDirection; }

	KRR_DEVICE bool isDeltaLight() const { return true; }

protected:
	KRR_CALLABLE Spectrum I(const Vector3f& w, const SampledWavelengths& lambda) const {
		return Spectrum::fromRGB(Iemit, SpectrumType::RGBIlluminant, lambda, *colorSpace) * scale
			* smooth_step(fabs(w.z()), cosOuterCone, cosInnerCone);
	}

	RGB Iemit;
	float scale;
	float cosInnerCone, cosOuterCone;
	Transformation transform;
	const RGBColorSpace *colorSpace;
};

class DiffuseAreaLight {
public:
	DiffuseAreaLight() = default;

	DiffuseAreaLight(const Shape &shape, Vector3f Le, bool twoSided = false, float scale = 1.f,
					 const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		shape(shape), Le(Le), twoSided(twoSided), scale(scale), colorSpace(colorSpace) {}

	DiffuseAreaLight(const Shape &shape, const rt::TextureData &texture, RGB Le = {},
					 bool twoSided = false, float scale = 1.f,
					 const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		shape(shape), texture(texture), Le(Le), twoSided(twoSided), scale(scale), colorSpace(colorSpace) {}

	DiffuseAreaLight(const Shape &shape, const rt::MaterialData *material) :
		shape(shape), material(material), twoSided(true), colorSpace(nullptr) {}

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const;

	template <bool Authored = true>
	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx,
													  const SampledWavelengths &lambda) const {
		LightSample ls				= {};
		ShapeSampleContext shapeCtx = {ctx.p, ctx.n};
		ShapeSample ss				= shape.sample(u, shapeCtx);
		DCHECK(!isnan(ss.pdf));
		Interaction &intr = ss.intr;
		intr.wo			  = normalize(ctx.p - intr.p);

		ls.intr = intr;
		ls.pdf	= ss.pdf;
		ls.L	= L<Authored>(intr.p, intr.n, intr.uv, intr.wo, lambda, true);
		return ls;
	}

	template <bool Authored = true>
	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
							  const SampledWavelengths &lambda, bool sampleOpacity = false) const {
		if (!twoSided && dot(n, w) < 0.f) return Spectrum::Zero(); // hit backface
		if constexpr (Authored) {
			if (material && material->mProgram.enabled)
				return evaluateEmission(p, n, w, lambda, sampleOpacity);
		}

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
	KRR_DEVICE KRR_NOINLINE Spectrum evaluateEmission(Vector3f p, Vector3f n, Vector3f w,
		const SampledWavelengths &lambda, bool sampleOpacity) const {
		MaterialContext context = shape.materialContext(p);
		auto values = material->mProgram.evaluate(context, MaterialEvaluation::Emission);
		if (values[MaterialParameter::ThinWalled][0] == 0 && dot(n, w) < 0) return Spectrum(0);
		RGB color = materialVector(values[MaterialParameter::EmissionColor]);
		float intensity = materialEmissionIntensity(values[MaterialParameter::EmissionLuminance][0],
			material->mProgram.model);
		if (sampleOpacity) {
			auto opacity = material->mProgram.evaluate(context, MaterialEvaluation::Opacity);
			intensity *= fminf(fmaxf(opacity[MaterialParameter::Opacity][0], 0), 1);
		}
		Spectrum emitted = intensity * Spectrum::fromRGB(color, SpectrumType::RGBIlluminant,
			lambda, *material->getColorSpace());
		if (values[MaterialParameter::CoatWeight][0] > 0 || values[MaterialParameter::FuzzWeight][0] > 0) {
			openpbr::Model model(values, context, w, lambda, *material->getColorSpace(), material->mProgram.model);
			emitted *= model.emissionScale();
		}
		return emitted;
	}

	Shape shape;
	rt::TextureData texture{}; // emissive image texture
	const rt::MaterialData *material{nullptr};
	RGB Le{0};
	bool twoSided{true};
	float scale{1};
	const RGBColorSpace *colorSpace;
};

class InfiniteLight {
public:
	InfiniteLight() = default;
	void release() noexcept { image.release(); }

	InfiniteLight(const Matrix3f &rotation, RGB tint, float scale = 1, float sceneRadius = 1e5f,
				  const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		tint(tint), scale(scale), rotation(rotation), sceneRadius(sceneRadius), colorSpace(colorSpace) {}

	InfiniteLight(const Matrix3f &rotation, const rt::TextureData &image, float scale = 1,
				  float sceneRadius = 1e5f, const RGBColorSpace *colorSpace = KRR_DEFAULT_COLORSPACE) :
		image(image), tint(RGB::Ones()), scale(scale), rotation(rotation), sceneRadius(sceneRadius), colorSpace(colorSpace) {}

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const;

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
		Vector2f uv = utils::worldToLatLong(rotation.transpose() * wi);
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
	public TaggedPointer<rt::PointLight, rt::DirectionalLight, rt::SpotLight, 
						rt::DiffuseAreaLight, rt::InfiniteLight> {
public:
	using TaggedPointer::TaggedPointer;

	template <bool Authored = true>
	KRR_DEVICE LightSample sampleLi(Vector2f u, const LightSampleContext &ctx, 
									const SampledWavelengths& lambda) const {
		auto sampleLi = [&](auto ptr) -> LightSample {
			using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
			if constexpr (std::is_same_v<T, DiffuseAreaLight>)
				return ptr->template sampleLi<Authored>(u, ctx, lambda);
			else return ptr->sampleLi(u, ctx, lambda);
		};
		return dispatch(sampleLi);
	}

	template <bool Authored = true>
	KRR_DEVICE Spectrum L(Vector3f p, Vector3f n, Vector2f uv, Vector3f w,
					   const SampledWavelengths &lambda) const {
		auto L = [&](auto ptr) -> Spectrum {
			using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
			if constexpr (std::is_same_v<T, DiffuseAreaLight>)
				return ptr->template L<Authored>(p, n, uv, w, lambda);
			else return ptr->L(p, n, uv, w, lambda);
		};
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
