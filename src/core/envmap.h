#pragma once

#include "common.h"
#include "math/math.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

struct LightSample {
	vec3f Li;
	float pdf;
	vec3f wi;
};

class EnvLight{
public:
	using SharedPtr = std::shared_ptr<EnvLight>;
	struct EnvLightData {
		vec3f tint = { 1,1,1 };
		float intensity = 1.0;
		float rotation = 0;
		Texture mEnvTexture;
	};

	__both__ void setRotation(float angle) { mData.rotation = angle; }

	void renderUI() {};
	bool update() {
		bool hasChanges = false;
		hasChanges |= mData.tint != mDataPrev.tint;
		hasChanges |= mData.intensity != mDataPrev.intensity;
		hasChanges |= mData.rotation != mDataPrev.rotation;
		mDataPrev = mData;
		return hasChanges;
	}

	__both__ void sample(LightSample& ls) {}
	__both__ void eval(LightSample& ls) {
		ls.pdf = 0.25 * M_1_PI;
		ls.Li = mData.tint * mData.intensity;
	}

private:
	EnvLightData mData, mDataPrev;
};

KRR_NAMESPACE_END