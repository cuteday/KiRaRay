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

	__both__ void setRotation(float angle) { mRotation = angle; }

	void renderUI() {};

	__both__ void sample(LightSample& ls) {}
	__both__ void eval(LightSample& ls) {
		ls.pdf = 0.25 * M_1_PI;
		ls.Li = mTint * mIntensity;
	}

private:
	vec3f mTint = { 1,1,1 };
	float mIntensity = 1.0;
	float mRotation = 0;
};

KRR_NAMESPACE_END