// This file is deprecated, EnvLight is moved to InfiniteLight, and will be deleted in the future.
#pragma once

#include "common.h"
#include "math/math.h"
#include "math/utils.h"
#include "texture.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

struct EnvLightSample {
	Color L;
	float pdf;
	Vector3f wi;
};

class EnvLight{
public:
	using SharedPtr = std::shared_ptr<EnvLight>;
	struct EnvLightData {
		Color tint		= Color::Ones();
		float intensity = 1.0;
		float rotation = 0;
		Texture mEnvTexture;
	};

	__both__ void setRotation(float angle) { mData.rotation = angle; }
	
	void setImage(const string &filename) {
		logDebug("Loading environment texture from: " + filename);
		mData.mEnvTexture.loadImage(filename);
		mData.mEnvTexture.toDevice();
	}

	void renderUI() {
		ui::SliderFloat("Intensity", &mData.intensity, 0, 10, "%.02f");
		ui::SliderFloat("Rotation", &mData.rotation, 0, 1, "%.03f");
		ui::ColorEdit3("Tint", (float*)&mData.tint);
		//ui::ColorPicker3("Tint", (float*)&mData.tint);
		if (mData.mEnvTexture.isOnDevice())
			ui::Checkbox("IBL", &mIBL);
	}

	bool update() {
		bool hasChanges = false;
		hasChanges |= any(mData.tint != mDataPrev.tint);
		hasChanges |= mData.intensity != mDataPrev.intensity;
		hasChanges |= mData.rotation != mDataPrev.rotation;
		mDataPrev = mData;
		return hasChanges;
	}

	__device__ float pdf(Vector3f wi) {
		return 0.25 * M_INV_PI;
	}

	__device__ EnvLightSample sample(Vector2f u) {
		Vector3f wi = utils::latlongToWorld(u);
		EnvLightSample ls = {};
		ls.wi = wi;
		ls.L = eval(wi);
		ls.pdf = 0.25 * M_INV_PI;
		return ls;
	}

	__device__ Color eval(Vector3f wi) {
		Color Li;
		Li = mData.tint * mData.intensity;

		//cudaTextureObject_t texture = mData.mEnvTexture.getCudaTexture();
		if (!mIBL && mData.mEnvTexture.isOnDevice()) return Li;
		Vector2f uv = utils::worldToLatLong(wi);
		uv[0] = fmod(uv[0] + mData.rotation, 1.f);
		Color env = mData.mEnvTexture.tex(uv);
		Li *= env;

		return Li;
	}

private:
	EnvLightData mData, mDataPrev;
	bool mIBL{ true };
};

KRR_NAMESPACE_END