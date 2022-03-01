#pragma once

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

class Material;
class Light;

struct Interaction{
	Interaction() = default;

	__both__ Interaction(vec3f p) : p(p){}

	__both__ Interaction(vec3f p, vec2f uv): p(p), uv(uv) {}

	__both__ Interaction(vec3f p, vec3f n, vec2f uv) : p(p), n(n), uv(uv) {}

	__both__ Interaction(vec3f p, vec3f wo, vec3f n, vec2f uv): p(p), wo(wo), n(n), uv(uv) {}

	__both__ inline vec3f offsetRayOrigin() {
		return utils::offsetRayOrigin(p, n, wo);
	}

	__both__ inline vec3f offsetRayOrigin(vec3f& w) {
		return utils::offsetRayOrigin(p, n, w);
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