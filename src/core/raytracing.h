#pragma once

#include "common.h"
#include "krrmath/vector.h"
#include "krrmath/transform.h"
#include "util/math_utils.h"
#include "taggedptr.h"
#include "medium.h"

#define KRR_RAY_TMAX	(1e20f)
#define KRR_RAY_EPS		(1e-5f)

KRR_NAMESPACE_BEGIN

namespace rt {
class MeshData;
class MaterialData;
class InstanceData;
}

typedef struct {
	rt::InstanceData *instance;
} HitgroupSBTData;

enum class MaterialType {
	Diffuse = 0,
	Dielectric,
	Disney,
	Count
};

class Ray {
public:
	KRR_CALLABLE Ray() = default;

	KRR_CALLABLE Ray(Vector3f o, Vector3f d, float time = 0, Medium medium = nullptr) :
		origin(o), dir(d), time(time), medium(medium) {}
	
	KRR_CALLABLE bool hasNaN() const { return origin.hasNaN() || dir.hasNaN(); }

	KRR_CALLABLE Vector3f operator()(float t) const { return origin + dir * t; }

	Vector3f origin;
	Vector3f dir;
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

KRR_CALLABLE Vector3f
offsetRayOrigin(const Vector3f &p, const Vector3f &n, const Vector3f &w) {
	Vector3f offset = n * KRR_RAY_EPS;
	if (dot(n, w) < 0.f) offset = -offset;
	Vector3f po = p + offset;
	return po;
}

KRR_CALLABLE Ray operator*(const Affine3f& transform, const Ray& ray) {
	Vector3f o = transform * ray.origin;
	Vector3f d = transform.rotation() * ray.dir;
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

	// spawn a ray from and to 2 slightly offseted points, length of direction is the distance
	KRR_CALLABLE Ray spawnRay(const Vector3f& to) const {
		Vector3f p_o = offsetRayOrigin(to - p);
		return { p_o, to - p_o };
	}

	KRR_CALLABLE Ray spawnRay(const Interaction& intr) const{
		Vector3f to = intr.offsetRayOrigin(p - intr.p);
		return spawnRay(to);
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

static constexpr int nSpectrumSamples = 4;

class SampledChannel {
public:
	KRR_CALLABLE SampledChannel() = default;

	KRR_CALLABLE SampledChannel(float u) : channel(u * Color3f::dim){}

	KRR_CALLABLE static SampledChannel sampleUniform(float u) { return SampledChannel(u); }

	KRR_CALLABLE operator int() const { return channel; }

	int channel;
};

KRR_NAMESPACE_END