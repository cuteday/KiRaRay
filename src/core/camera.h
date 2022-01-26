#pragma once

#include "common.h" 
#include "io.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

using namespace math;

class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;

	Camera() {}

	__both__ vec3f getRayDir(vec2i pixel, vec2i frameSize) {

		vec2f ndc = vec2f(2, -2) * (vec2f(pixel) + vec2f (0.5)) / vec2f(frameSize) + vec2f(-1, 1);
		return	normalize(ndc.x * mData.u + ndc.y * mData.v + mData.w);
	}	

	__both__ vec3f getAspectRatio() { return mData.aspectRatio; }
	__both__ vec3f getPosition() { return mData.pos; }
	__both__ vec3f getTarget() { return mData.target; }
	__both__ vec3f getForward() { return normalize(mData.target - mData.pos); }
	__both__ vec3f getUp() { return normalize(mData.up); }
	__both__ vec3f getRight() { return cross(getForward(), getUp()); }

	__both__ void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	__both__ void setPosition(vec3f& pos) { mData.pos = pos; }
	__both__ void setTarget(vec3f& target) { mData.target = target; }
	__both__ void setUp(vec3f& up) { mData.up = up; }

protected:

	struct{
		vec3f pos = { 0, 0, 0 };
		vec3f target = { 0, 0, -1 };
		vec3f up = { 0, 1, 0 };

		vec3f u = { 1, 0, 0 };		// camera right		[dependent to aspect ratio]
		vec3f v = { 0, 1, 0 };		// camera up		[dependent to aspect ratio]
		vec3f w = { 0, 0, -1 };		// camera forward

		float aspectRatio = 1.777777f;
	} mData;

};

class CameraController{
public:
	using SharedPtr = std::shared_ptr<CameraController>;

	virtual void update() = 0;
	virtual void onMouseEvent(io::MouseEvent& mouseEvent) = 0;
	virtual void onKeyEvent(io::KeyboardEvent& keyEvent) = 0;

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
	virtual void onMouseEvent(io::MouseEvent& mouseEvent) override;
	virtual void onKeyEvent(io::KeyboardEvent& keyEvent) override;

private:
	struct {
		vec3f target = { 0, 0, 0 };
		vec3f up = { 0, 1, 0 };
		float radius = 5;

		float dampling = 1;
	}mData;
};

KRR_NAMESPACE_END