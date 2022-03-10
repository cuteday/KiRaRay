#pragma once

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

class Material;
class Light;

struct Ray {
	vec3f origin;
	vec3f dir;
};

struct Interaction{
	Interaction() = default;

	__both__ Interaction(vec3f p) : p(p){}

	__both__ Interaction(vec3f p, vec2f uv): p(p), uv(uv) {}

	__both__ Interaction(vec3f p, vec3f n, vec2f uv) : p(p), n(n), uv(uv) {}

	__both__ Interaction(vec3f p, vec3f wo, vec3f n, vec2f uv): p(p), wo(wo), n(n), uv(uv) {}

	KRR_CALLABLE vec3f offsetRayOrigin(const vec3f& w) const {
		return utils::offsetRayOrigin(p, n, w);
	}

	KRR_CALLABLE vec3f offsetRayOrigin() {
		return offsetRayOrigin(wo);
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

struct SurfaceInteraction : Interaction {


	Material* material{ nullptr };
	Light* areaLight{ nullptr };
};

KRR_NAMESPACE_END