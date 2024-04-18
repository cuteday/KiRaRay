#pragma once

#include "common.h"
#include "krrmath/vector.h"
#include "krrmath/transform.h"
#include "util/math_utils.h"
#include "device/taggedptr.h"
#include "medium.h"

#define KRR_RAY_EPS		(1e-4f)

NAMESPACE_BEGIN(krr)

namespace rt {
class MeshData;
class MaterialData;
class InstanceData;
}

typedef struct {
	rt::InstanceData *instance;
} HitgroupSBTData;

enum class MaterialType : uint32_t{
	Null = 0,
	Diffuse,
	Dielectric,
	Conductor,
	Disney,
	Count
};

KRR_ENUM_DEFINE(MaterialType, {
	{MaterialType::Null, "null"},
	{MaterialType::Diffuse, "diffuse"},
	{MaterialType::Conductor, "conductor"},
	{MaterialType::Dielectric, "dielectric"},
	{MaterialType::Disney, "disney"}
})

class Ray {
public:
	KRR_CALLABLE Ray() = default;

	KRR_CALLABLE Ray(Vector3f o, Vector3f d, float time = 0, Medium medium = nullptr) :
		origin(o), dir(d), time(time), medium(medium) {}
	
	KRR_CALLABLE bool hasNaN() const { return origin.hasNaN() || dir.hasNaN(); }

	KRR_CALLABLE Vector3f operator()(float t) const { return origin + dir * t; }

	Vector3f origin{};
	Vector3f dir{};
	float time{0};
	Medium medium{nullptr};
};

struct Frame {
	Frame() = default;

	KRR_CALLABLE Frame(Vector3f n, Vector3f t, Vector3f b) : N(n), T(t), B(b) {}

	KRR_CALLABLE Frame(Vector3f n) : N(n) {
		T = utils::getPerpendicular(n);
		B = normalize(cross(n, T));
	}

	KRR_CALLABLE Vector3f toWorld(const Vector3f& v) const {
		return T * v[0] + B * v[1] + N * v[2];
	}

	KRR_CALLABLE Vector3f toLocal(const Vector3f& v) const {
		return { dot(T, v), dot(B, v), dot(N, v) };
	}

	Vector3f N;
	Vector3f T;
	Vector3f B;
};

class Transformation {
public:
	Transformation() = default;

	KRR_CALLABLE Transformation(const Matrix4f& m) : m(m), mInv(m.inverse()) {}
	KRR_CALLABLE Transformation(const Affine3f& m) : m(m), mInv(m.matrix().inverse()) {}
	KRR_CALLABLE Transformation(const Matrix4f& m, const Matrix4f& mInv) : m(m), mInv(mInv) {}
	KRR_CALLABLE Transformation(const Affine3f& m, const Affine3f& mInv) : m(m), mInv(mInv) {}

	KRR_CALLABLE Vector3f translation() const { return m.translation(); }
	KRR_CALLABLE Matrix3f rotation() const {return m.linear(); }
	KRR_CALLABLE Matrix4f matrix() const { return m.matrix(); }
	KRR_CALLABLE Affine3f transform() const { return m; } 
	KRR_CALLABLE Affine3f inverse() const { return mInv; }
	KRR_CALLABLE Matrix3f transposedInverse() const {
		return mInv.matrix().block<3, 3>(0, 0).transpose();
	}

	KRR_CALLABLE Ray operator()(const Ray& ray) const {
		return Ray{m * ray.origin, m.matrix().block<3, 3>(0, 0) * ray.dir, ray.time, ray.medium};
	}

	Affine3f m, mInv;
};

KRR_CALLABLE Vector3f
offsetRayOrigin(const Vector3f &p, const Vector3f &n, const Vector3f &w) {
	Vector3f offset = n * KRR_RAY_EPS;
	if (dot(n, w) < 0.f) offset = -offset;
	return p + offset;
}

KRR_CALLABLE Ray operator*(const Affine3f& transform, const Ray& ray) {
	Vector3f o = transform * ray.origin;
	Vector3f d = transform.matrix().block<3, 3>(0, 0) * ray.dir;
	return Ray{o, d, ray.time, ray.medium};
}

struct Interaction{
	Interaction() = default;

	KRR_CALLABLE Interaction(Vector3f p) : p(p){}
	KRR_CALLABLE Interaction(Vector3f p, Vector2f uv): p(p), uv(uv) {}
	KRR_CALLABLE Interaction(Vector3f p, Vector3f n, Vector2f uv) : p(p), n(n), uv(uv) {}
	KRR_CALLABLE Interaction(Vector3f p, Vector3f wo, Vector3f n, Vector2f uv): p(p), wo(wo), n(n), uv(uv) {}
	
	KRR_CALLABLE Interaction(Vector3f p, Vector3f wo, float time, Medium medium): p(p), wo(wo), time(time), medium(medium) {}
	KRR_CALLABLE Interaction(Vector3f p, float time, Medium medium): p(p), wo(wo), medium(medium) {}
	KRR_CALLABLE Interaction(Vector3f p, const MediumInterface *mediumInterface): p(p), mediumInterface(mediumInterface) {}
	KRR_CALLABLE Interaction(Vector3f p, float time, const MediumInterface *mediumInterface): p(p), time(time), mediumInterface(mediumInterface) {}
	KRR_CALLABLE Interaction(Vector3f p, Vector3f wo, Vector3f n, Vector2f uv, float time): p(p), wo(wo), n(n), uv(uv), time(time) {}

	KRR_CALLABLE bool isSurfaceInteraction() const { return !n.isZero(); }
	KRR_CALLABLE bool isMediumInteraction() const { return n.isZero(); }

	KRR_CALLABLE Vector3f offsetRayOrigin(const Vector3f& w) const {
		return krr::offsetRayOrigin(p, n, w);
	}

	KRR_CALLABLE Vector3f offsetRayOrigin() {
		return Interaction::offsetRayOrigin(wo);
	}

	/* spawn a ray towards a given direction, which is usually normalized. */
	KRR_CALLABLE Ray spawnRayTowards(const Vector3f& dir) const {
		return Ray{offsetRayOrigin(dir), dir, time, getMedium(dir)};
	}

	/* spawn a ray from and to 2 slightly offseted points,
		length of direction is the distance, useful for shadow rays. */
	KRR_CALLABLE Ray spawnRayTo(const Vector3f& to) const {
		Vector3f p_o = offsetRayOrigin(to - p);
		Vector3f d	 = to - p_o;
		return {p_o, d, time, getMedium(d)};
	}

	KRR_CALLABLE Ray spawnRayTo(const Interaction& intr) const{
		Vector3f to = intr.offsetRayOrigin(p - intr.p);
		return spawnRayTo(to);
	}

	KRR_CALLABLE Medium getMedium(Vector3f w) const {
		if(mediumInterface) 
			return w.dot(n) > 0 ? mediumInterface->outside : mediumInterface->inside;		
		return medium;
	}

	KRR_CALLABLE Medium getMedium() const {
		if (mediumInterface) DCHECK_EQ(mediumInterface->inside, mediumInterface->outside);
		return mediumInterface ? mediumInterface->inside : medium;
	}

	Vector3f p {0};
	Vector3f wo {0};	// world-space out-scattering direction
	Vector3f n {0};
	Vector2f uv {0};
	float time{0};
	const MediumInterface *mediumInterface{nullptr};
	Medium medium{nullptr};
};

NAMESPACE_END(krr)