#pragma once

#include "common.h" 
#include "math/math.h"

KRR_NAMESPACE_BEGIN

using namespace math;

class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;

	Camera() {}

	__both__ vec3f getRayDir(vec2i pixel, vec2i frameSize) {

		vec2f ndc = vec2f(2, -2) * (vec2f(pixel) + vec2f (0.5)) / vec2f(frameSize) + vec2f(-1, 1);
		return	normalize(ndc.x * cameraU + ndc.y * cameraV + cameraW);
	}	

	vec3f pos = { 0, 0, 0 };
	vec3f target = { 0, 0, -1 };
	vec3f up = { 0, 1, 0 };

	vec3f cameraU = { 1, 0, 0 };		// camera right		[dependent to aspect ratio]
	vec3f cameraV = { 0, 1, 0 };		// camera up		[dependent to aspect ratio]
	vec3f cameraW = { 0, 0, -1 };		// camera forward

	float aspectRatio = 1.777777f;
	
};

class CameraController{
public:

	virtual void onMouseEvent(io::MouseEvent& mouseEvent) {}
    virtual void onKeyEvent(io::KeyboardEvent &keyEvent) {}

private:
	Camera::SharedPtr mpCamera;
};

KRR_NAMESPACE_END