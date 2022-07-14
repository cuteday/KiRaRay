#pragma once

#include "common.h"

#include "math/math.h"
#include "math/utils.h"

#define KRR_RAY_TMAX	(1e20f)
#define KRR_RAY_EPS		(1e-4f)

KRR_NAMESPACE_BEGIN

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
	vec3f origin;
	vec3f dir;
};

class RayDifferential : public Ray {
public:
	bool hasDifferentials{ false };
	vec3f rxOrigin, ryOrigin;
	vec3f rxDir, ryDir;
};

KRR_CALLABLE vec3f offsetRayOrigin(vec3f p, vec3f n, vec3f w) {
	vec3f offset = n * KRR_RAY_EPS;
	if (dot(n, w) < 0.f)
		offset = -offset;
	return p + offset;
}

struct Frame {
	Frame() = default;

	Frame(vec3f n, vec3f t, vec3f b) : N(n), T(t), B(b) {}

	Frame(vec3f n) : N(n) {
		T = math::utils::getPerpendicular(N);
		B = normalize(cross(N, T));
	}

	KRR_CALLABLE vec3f toWorld(vec3f v) const {
		return T * v[0] + B * v[1] + N * v[2];
	}

	KRR_CALLABLE vec3f toLocal(vec3f v) const {
		return { dot(T, v), dot(B, v), dot(N, v) };
	}

	vec3f N, T, B;
};

struct Interaction{
	Interaction() = default;

	KRR_CALLABLE Interaction(vec3f p) : p(p){}
	KRR_CALLABLE Interaction(vec3f p, vec2f uv): p(p), uv(uv) {}
	KRR_CALLABLE Interaction(vec3f p, vec3f n, vec2f uv) : p(p), n(n), uv(uv) {}
	KRR_CALLABLE Interaction(vec3f p, vec3f wo, vec3f n, vec2f uv): p(p), wo(wo), n(n), uv(uv) {}

	KRR_CALLABLE vec3f offsetRayOrigin(const vec3f& w) const {
		return krr::offsetRayOrigin(p, n, w);
	}

	KRR_CALLABLE vec3f offsetRayOrigin() {
		return Interaction::offsetRayOrigin(wo);
	}

	// spawn a ray from and to 2 slightly offseted points, length of direction is the distance
	KRR_CALLABLE Ray spawnRay(const vec3f& to) const {
		vec3f p_o = offsetRayOrigin(to - p);
		return { p_o, to - p_o };
	}

	KRR_CALLABLE Ray spawnRay(const Interaction& intr) const{
		vec3f to = intr.offsetRayOrigin(p - intr.p);
		return spawnRay(to);
	}

	vec3f p {0};
	vec3f wo {0};	// world-space out-scattering direction
	vec3f n {0};
	vec2f uv {0};
};

KRR_NAMESPACE_END