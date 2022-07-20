#pragma once

#include "common.h"

#include "math/math.h"
#include "math/utils.h"

#define KRR_RAY_TMAX	(1e20f)
#define KRR_RAY_EPS		(1e-4f)

KRR_NAMESPACE_BEGIN

using namespace math;

class Material;
class Light;

enum class BsdfType {
	Diffuse = 0,
	FresnelBlend,
	Disney,
	Count
};

class Ray {
public:
	Vec3f origin;
	Vec3f dir;
};

class RayDifferential : public Ray {
public:
	bool hasDifferentials{ false };
	Vec3f rxOrigin, ryOrigin;
	Vec3f rxDir, ryDir;
};

KRR_CALLABLE Vec3f offsetRayOrigin(Vec3f p, Vec3f n, Vec3f w) {
	Vec3f offset = n * KRR_RAY_EPS;
	if (dot(n, w) < 0.f)
		offset = -offset;
	return p + offset;
}

struct Frame {
	Frame() = default;

	Frame(Vec3f n, Vec3f t, Vec3f b) : N(n), T(t), B(b) {}

	Frame(Vec3f n) : N(n) {
		T = math::utils::getPerpendicular(N);
		B = normalize(cross(N, T));
	}

	KRR_CALLABLE Vec3f toWorld(Vec3f v) const {
		return T * v[0] + B * v[1] + N * v[2];
	}

	KRR_CALLABLE Vec3f toLocal(Vec3f v) const {
		return { dot(T, v), dot(B, v), dot(N, v) };
	}

	Vec3f N, T, B;
};

struct Interaction{
	Interaction() = default;

	KRR_CALLABLE Interaction(Vec3f p) : p(p){}
	KRR_CALLABLE Interaction(Vec3f p, Vec2f uv): p(p), uv(uv) {}
	KRR_CALLABLE Interaction(Vec3f p, Vec3f n, Vec2f uv) : p(p), n(n), uv(uv) {}
	KRR_CALLABLE Interaction(Vec3f p, Vec3f wo, Vec3f n, Vec2f uv): p(p), wo(wo), n(n), uv(uv) {}

	KRR_CALLABLE Vec3f offsetRayOrigin(const Vec3f& w) const {
		return krr::offsetRayOrigin(p, n, w);
	}

	KRR_CALLABLE Vec3f offsetRayOrigin() {
		return Interaction::offsetRayOrigin(wo);
	}

	// spawn a ray from and to 2 slightly offseted points, length of direction is the distance
	KRR_CALLABLE Ray spawnRay(const Vec3f& to) const {
		Vec3f p_o = offsetRayOrigin(to - p);
		return { p_o, to - p_o };
	}

	KRR_CALLABLE Ray spawnRay(const Interaction& intr) const{
		Vec3f to = intr.offsetRayOrigin(p - intr.p);
		return spawnRay(to);
	}

	Vec3f p {0};
	Vec3f wo {0};	// world-space out-scattering direction
	Vec3f n {0};
	Vec2f uv {0};
};

KRR_NAMESPACE_END