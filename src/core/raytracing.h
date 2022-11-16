#pragma once

#include "common.h"

#include "math/math.h"
#include "math/utils.h"

#define KRR_RAY_TMAX	(1e20f)
#define KRR_RAY_EPS		(1e-5f)

KRR_NAMESPACE_BEGIN

using namespace math;

class Material;
class Light;

enum class MaterialType {
	Diffuse = 0,
	FresnelBlend,
	Disney,
	Count
};

class Ray {
public:
	Vector3f origin;
	Vector3f dir;
};

class RayDifferential : public Ray {
public:
	bool hasDifferentials{ false };
	Vector3f rxOrigin, ryOrigin;
	Vector3f rxDir, ryDir;
};

KRR_CALLABLE Vector3f offsetRayOrigin(Vector3f p, Vector3f n, Vector3f w) {
	Vector3f offset = n * KRR_RAY_EPS;
	if (dot(n, w) < 0.f)
		offset = -offset;
	Vector3f po = p + offset;
	//for (int i = 0; i < Vector3f::dim; i++) {
	//	if (offset[i] > 0)
	//		po[i] = utils::nextFloatUp(po[i]);
	//	else if (offset[i] < 0)
	//		po[i] = utils::nextFloatDown(po[i]);
	//}
	return po;
}

struct Frame {
	Frame() = default;

	KRR_CALLABLE Frame(Vector3f n, Vector3f t, Vector3f b) : N(n), T(t), B(b) {}

	KRR_CALLABLE Frame(Vector3f n) : N(n) {
		T = math::utils::getPerpendicular(n);
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

struct Interaction{
	Interaction() = default;

	KRR_CALLABLE Interaction(Vector3f p) : p(p){}
	KRR_CALLABLE Interaction(Vector3f p, Vector2f uv): p(p), uv(uv) {}
	KRR_CALLABLE Interaction(Vector3f p, Vector3f n, Vector2f uv) : p(p), n(n), uv(uv) {}
	KRR_CALLABLE Interaction(Vector3f p, Vector3f wo, Vector3f n, Vector2f uv): p(p), wo(wo), n(n), uv(uv) {}

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

	Vector3f p {0};
	Vector3f wo {0};	// world-space out-scattering direction
	Vector3f n {0};
	Vector2f uv {0};
};

KRR_NAMESPACE_END