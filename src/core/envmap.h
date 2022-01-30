#pragma once

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

struct LightSample {
	vec3f Li;
	float pdf;
	vec3f wi;
};

class EnvLight{
public:
	using SharedPtr = std::shared_ptr<EnvLight>;

	virtual void setRotation(float angle) { rotation = angle; }

	__both__ void sample(LightSample& ls) {}
	__both__ void eval(LightSample& ls) {
		ls.pdf = 0.25 / M_PI;
		ls.Li = color * intensity;
	}


private:
	vec3f color = { 1,1,1 };
	float intensity = 1;
	float rotation = 0;
};

KRR_NAMESPACE_END