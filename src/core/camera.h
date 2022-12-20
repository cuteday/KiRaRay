#pragma once

#include "kiraray.h"
#include "raytracing.h"
#include "render/sampling.h"
#include "sampler.h"
#include "io.h"

KRR_NAMESPACE_BEGIN

using namespace io;


class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;

	struct CameraData {
		Vector2f filmSize{ 42.666667f, 24.0f };	// sensor size in mm [width, height]
		float focalLength{ 21 };				// distance from sensor to lens, in mm
		float focalDistance{ 10 };				// distance from length to focal point, in scene units (m)
		float lensRadius{ 0 };					// aperture radius, in mm
		float aspectRatio{ 1.777777f };			// width divides height

		Vector3f pos{ 0, 0, 0 };
		Vector3f target{ 0, 0, -1 };
		Vector3f up{ 0, 1, 0 };

		Vector3f u{ 1, 0, 0 };					// camera right		[dependent to aspect ratio]
		Vector3f v{ 0, 1, 0 };					// camera up		[dependent to aspect ratio]
		Vector3f w{ 0, 0, -1 };					// camera forward
	
		KRR_CLASS_DEFINE(CameraData, pos, target, up, focalLength, focalDistance, lensRadius, aspectRatio);
	};

	Camera() = default;

	KRR_CALLABLE Ray getRay(Vector2i pixel, Vector2i frameSize, Sampler& sampler) {
		Ray ray;
		/* 1. Statified sample on the film plane (within the fragment) */
		Vector2f p = (Vector2f)pixel + Vector2f(0.5f) + sampler.get2D(); // uniform sample + box filter
		Vector2f ndc = Vector2f(2 * p) / Vector2f(frameSize) + Vector2f(-1.f); // ndc in [-1, 1]^2
		if (mData.lensRadius > 0) {			/*Thin lens*/
			/* 2. Sample the lens (uniform) */
			Vector3f focalPoint = mData.pos + ndc[0] * mData.u + ndc[1] * mData.v + mData.w;
			Vector2f apertureSample = mData.lensRadius > M_EPSILON ? uniformSampleDisk(sampler.get2D()) : Vector2f::Zero();
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

	KRR_CALLABLE float getAspectRatio() const { return mData.aspectRatio; }
	KRR_CALLABLE Vector3f getPosition() const { return mData.pos; }
	KRR_CALLABLE Vector3f getTarget() const { return mData.target; }
	KRR_CALLABLE Vector3f getForward() const { return normalize(mData.target - mData.pos); }
	KRR_CALLABLE Vector3f getUp() const { return normalize(mData.v); }
	KRR_CALLABLE Vector3f getRight() const { return normalize(mData.u); }
	KRR_CALLABLE Vector2f getFilmSize() const { return mData.filmSize; }
	KRR_CALLABLE float getfocalDistance() const { return mData.focalDistance; }
	KRR_CALLABLE float getfocalLength() const { return mData.focalLength; }
	KRR_CALLABLE CameraData getData() const { return mData; }

	KRR_CALLABLE void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	KRR_CALLABLE void setFilmSize(Vector2f& size) { mData.filmSize = size; }
	KRR_CALLABLE void setfocalDistance(float focalDistance) { mData.focalDistance = focalDistance; }
	KRR_CALLABLE void setfocalDLength(float focalLength) { mData.focalLength = focalLength; }
	KRR_CALLABLE void setPosition(Vector3f& pos) { mData.pos = pos; }
	KRR_CALLABLE void setTarget(Vector3f& target) { mData.target = target; }
	KRR_CALLABLE void setUp(Vector3f& up) { mData.up = up; }

protected:
	KRR_CLASS_DEFINE(Camera, mData);
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
	virtual void setCamera(const Camera::SharedPtr &pCamera) { mpCamera = pCamera; }
		
protected:
	CameraController() = default;
	CameraController(const Camera::SharedPtr& pCamera){ 
		setCamera(pCamera);
	}

	Camera::SharedPtr mpCamera{};
	float mSpeed = 1.f;
};

class OrbitCameraController: public CameraController{
public:
	using SharedPtr = std::shared_ptr<OrbitCameraController>;
	struct CameraControllerData{
		Vector3f target = { 0, 0, 0 };
		Vector3f up = { 0, 1, 0 };
		float radius = 5;
		float pitch = 0;
		float yaw = 0;
		
		KRR_CLASS_DEFINE(CameraControllerData, radius, pitch, yaw, target);
	};

	OrbitCameraController() = default;
	OrbitCameraController(Camera::SharedPtr pCamera): CameraController(pCamera) {}
	
	static SharedPtr create(Camera::SharedPtr pCamera){
		return SharedPtr(new OrbitCameraController(pCamera));
	}
	
	virtual bool update() override;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;
	virtual void renderUI() override;
	virtual void setCamera(const Camera::SharedPtr &pCamera) override;

private:
	KRR_CLASS_DEFINE(OrbitCameraController, mData);
	CameraControllerData mData, mDataPrev;
	Vector2f mLastMousePos;
	float mDampling = 1;
	bool mOrbiting = false;
	bool mPanning = false;
	float mOrbitSpeed = 1;
	float mPanSpeed = 1;
	float mZoomSpeed = 0.1;
};

KRR_NAMESPACE_END