#pragma once

#include "kiraray.h"
#include "raytracing.h"
#include "render/sampling.h"
#include "sampler.h"
#include "io.h"

KRR_NAMESPACE_BEGIN

using namespace io;
using namespace math;

class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;

	struct CameraData {
		vec2f filmSize{ 42.666667f, 24.0f };	// sensor size in mm [width, height]
		float focalLength{ 21 };				// distance from sensor to lens, in mm
		float focalDistance{ 10 };			// distance from length to focal point, in scene units (m)
		float lensRadius{ 0 };					// aperture radius, in mm
		float aspectRatio{ 1.777777f };			// width divides height

		vec3f pos{ 0, 0, 0 };
		vec3f target{ 0, 0, -1 };
		vec3f up{ 0, 1, 0 };

		vec3f u{ 1, 0, 0 };					// camera right		[dependent to aspect ratio]
		vec3f v{ 0, 1, 0 };					// camera up		[dependent to aspect ratio]
		vec3f w{ 0, 0, -1 };				// camera forward
	};

	Camera() = default;

	KRR_CALLABLE Ray getRay(vec2i pixel, vec2i frameSize, Sampler sampler) {
		Ray ray;
		vec2f p = (vec2f)pixel + vec2f(0.5f) + sampler.get2D() /*uniform sample + box filter*/;
		vec2f ndc = vec2f(2 * p) / vec2f(frameSize) + vec2f(-1.f); // ndc in [-1, 1]^2
		if (mData.lensRadius > 0) {			/*Thin lens*/
			vec3f focalPoint = mData.pos + ndc[0] * mData.u + ndc[1] * mData.v + mData.w;
			vec2f apertureSample = mData.lensRadius > M_EPSILON ? uniformSampleDisk(sampler.get2D()) : vec2f::Zero();
			ray.origin = mData.pos + mData.lensRadius * (apertureSample[0] * normalize(mData.u) + apertureSample[1] * normalize(mData.v));
			ray.dir = normalize(focalPoint - ray.origin);
		} else {							/*Pin hole*/
			ray.origin = mData.pos;
			ray.dir = normalize(ndc[0] * mData.u + ndc[1] * mData.v + mData.w);
		}
		return ray;
	}

	bool update();
	void renderUI();

	KRR_CALLABLE float getAspectRatio() { return mData.aspectRatio; }
	KRR_CALLABLE vec3f getPosition() { return mData.pos; }
	KRR_CALLABLE vec3f getTarget() { return mData.target; }
	KRR_CALLABLE vec3f getForward() { return normalize(mData.target - mData.pos); }
	KRR_CALLABLE vec3f getUp() { return normalize(mData.v); }
	KRR_CALLABLE vec3f getRight() { return normalize(mData.u); }
	KRR_CALLABLE vec2f getFilmSize() { return mData.filmSize; }
	KRR_CALLABLE float getfocalDistance() { return mData.focalDistance; }
	KRR_CALLABLE float getfocalLength() { return mData.focalLength; }

	KRR_CALLABLE void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	KRR_CALLABLE void setFilmSize(vec2f& size) { mData.filmSize = size; }
	KRR_CALLABLE void setfocalDistance(float focalDistance) { mData.focalDistance = focalDistance; }
	KRR_CALLABLE void setfocalDLength(float focalLength) { mData.focalLength = focalLength; }
	KRR_CALLABLE void setPosition(vec3f& pos) { mData.pos = pos; }
	KRR_CALLABLE void setTarget(vec3f& target) { mData.target = target; }
	KRR_CALLABLE void setUp(vec3f& up) { mData.up = up; }

protected:
	CameraData mData, mDataPrev;
	bool mHasChanges = false;
	bool mPreserveHeight = true;	// preserve sensor height on aspect ratio changes.
};

class CameraController{
public:
	using SharedPtr = std::shared_ptr<CameraController>;

	virtual bool update() = 0;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) = 0;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) = 0;
	virtual void renderUI() {};
protected:
	CameraController(const Camera::SharedPtr& pCamera) : mpCamera(pCamera) {}

	Camera::SharedPtr mpCamera;
	float mSpeed = 1.f;
};

class OrbitCameraController: public CameraController{
public:
	using SharedPtr = std::shared_ptr<OrbitCameraController>;
	struct CameraControllerData{
		vec3f target = { 0, 0, 0 };
		vec3f up = { 0, 1, 0 };
		float radius = 5;
		float pitch = 0;
		float yaw = 0;
	};

	OrbitCameraController(Camera::SharedPtr pCamera): CameraController(pCamera) {}
	
	static SharedPtr create(Camera::SharedPtr pCamera){
		return SharedPtr(new OrbitCameraController(pCamera));
	}
	
	virtual bool update() override;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;
	virtual void renderUI() override;

private:
	CameraControllerData mData, mDataPrev;

	vec2f mLastMousePos;
	float mDampling = 1;
	bool mOrbiting = false;
	bool mPanning = false;
	float mOrbitSpeed = 1;
	float mPanSpeed = 1;
	float mZoomSpeed = 0.1;
};

KRR_NAMESPACE_END