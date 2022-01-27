#pragma once

#include "kiraray.h"
#include "io.h"

KRR_NAMESPACE_BEGIN

using namespace io;
using namespace math;

class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;

	Camera() {}

	__both__ vec3f getRayDir(vec2i pixel, vec2i frameSize) {

		vec2f ndc = vec2f(2, -2) * (vec2f(pixel) + vec2f (0.5)) / vec2f(frameSize) + vec2f(-1, 1);
		return	normalize(ndc.x * mData.u + ndc.y * mData.v + mData.w);
	}	

	//void beginFrame();
	void update();
	void renderUI() {}

	__both__ float getAspectRatio() { return mData.aspectRatio; }
	__both__ vec3f getPosition() { return mData.pos; }
	__both__ vec3f getTarget() { return mData.target; }
	__both__ vec3f getForward() { return normalize(mData.target - mData.pos); }
	__both__ vec3f getUp() { return normalize(mData.up); }
	__both__ vec3f getRight() { return normalize(mData.u); }
	__both__ vec2f getFrameSize() { return mData.frameSize; }
	__both__ float getFocalLength() { return mData.focalLength; }

	__both__ void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	__both__ void setFrameSize(vec2f size) { mData.frameSize = size; }
	__both__ void setPosition(vec3f& pos) { mData.pos = pos; }
	__both__ void setTarget(vec3f& target) { mData.target = target; }
	__both__ void setUp(vec3f& up) { mData.up = up; }

protected:

	struct{
		vec2f frameSize = { 42.666667f, 24.0f };	// sensor size in mm [width, height]
		float focalLength = 5;			// the distance from lens (pin hole) to sensor, in mm
		float aspectRatio = 1.777777f;	// width divides height

		vec3f pos = { 0, 0, 0 };
		vec3f target = { 0, 0, -1 };
		vec3f up = { 0, 1, 0 };

		vec3f u = { 1, 0, 0 };		// camera right		[dependent to aspect ratio]
		vec3f v = { 0, 1, 0 };		// camera up		[dependent to aspect ratio]
		vec3f w = { 0, 0, -1 };		// camera forward

		vec2f jitter = {0, 0};

	} mData;

	bool mPreserveHeight = true;	// preserve sensor height on aspect ratio changes.
};

class CameraController{
public:
	using SharedPtr = std::shared_ptr<CameraController>;

	virtual void update() = 0;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) = 0;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) = 0;

protected:
	CameraController(const Camera::SharedPtr& pCamera) : mpCamera(pCamera) {}

	Camera::SharedPtr mpCamera;
	float mSpeed = 1.f;
};

class OrbitCameraController: public CameraController{
public:
	using SharedPtr = std::shared_ptr<OrbitCameraController>;

	OrbitCameraController(Camera::SharedPtr pCamera): CameraController(pCamera) {}
	static SharedPtr create(Camera::SharedPtr pCamera){
		return SharedPtr(new OrbitCameraController(pCamera));
	}
	
	virtual void update() override;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;

private:
	struct {
		vec3f target = { 0, 0, 0 };
		vec3f up = { 0, 1, 0 };
		float radius = 5;

		float pitch = 0;
		float yaw = 0;

	}mData;

	// input states
	vec2f mLastMousePos;
	float mDampling = 1;
	bool mOrbiting = false;
	float mOrbitSpeed = 1;
	float mPanSpeed = 1;
};

KRR_NAMESPACE_END