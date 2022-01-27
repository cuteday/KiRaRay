#include "camera.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

void Camera::update()
{
	if (mPreserveHeight)
		mData.frameSize.x = mData.aspectRatio * mData.frameSize.y;
	else mData.frameSize.y = mData.frameSize.x / mData.aspectRatio;

	// vertical fov
	float fov = atan2(mData.frameSize.y * 0.5, mData.focalLength);
	
	mData.w = mData.focalLength * normalize(mData.target - mData.pos);
	mData.u = normalize(cross(mData.w, mData.up)) * mData.frameSize.x * 0.5f;
	mData.v = normalize(cross(mData.u, mData.w)) * mData.frameSize.y * 0.5f;
}

void Camera::renderUI()
{
	ui::Text("Hello from camera!");
	ui::Text("Position: %f, %f, %f", mData.pos.x, mData.pos.y, mData.pos.z);
	ui::Text("Target: %f, %f, %f", mData.target.x, mData.target.y, mData.target.z);

	ui::Text("Focal vector: %f, %f, %f", mData.w.x, mData.w.y, mData.w.z);
	ui::Text("Sensor right: %f, %f, %f", mData.u.x, mData.u.y, mData.u.z);
	ui::Text("Sensor up: %f, %f, %f", mData.v.x, mData.v.y, mData.v.z);
}

void OrbitCameraController::update()
{
	//quat yawRotate = quat::rotate(vec3f(0, 1, 0), mData.yaw);
	//quat pitchRotate = quat::rotate(vec3f(1, 0, 0), mData.pitch);
	//quat rotate = normalize(yawRotate * pitchRotate);
	quat rotate = normalize(quat(mData.yaw, mData.pitch, 0));
	vec3f forward = rotate * vec3f(0, 0, -1);
	vec3f pos = mData.target - forward * mData.radius;

	mpCamera->setPosition(pos);
	mpCamera->setTarget(mData.target);
	mpCamera->update();
}

bool OrbitCameraController::onMouseEvent(const MouseEvent& mouseEvent)
{

	switch (mouseEvent.type) {
	case io::MouseEvent::Type::Wheel:
		mData.radius -= mouseEvent.wheelDelta.y * 0.2f;
		return true;
	case io::MouseEvent::Type::LeftButtonDown:
		mOrbiting = true;
		return true;
	case io::MouseEvent::Type::LeftButtonUp:
		mOrbiting = false;
		return true;
	case io::MouseEvent::Type::Move:
		vec2f curMousePos = mouseEvent.pos;
		if(mOrbiting){
			vec2f deltaPos = curMousePos - mLastMousePos;
			mData.yaw -= deltaPos.x * mOrbitSpeed;
			mData.pitch -= deltaPos.y * mOrbitSpeed;
			mData.yaw = fmod(mData.yaw, 2 * M_PI);
			mData.pitch = clamp(mData.pitch, -M_PI / 2, M_PI / 2);
		}
		mLastMousePos = curMousePos;
	}
	return false;
}

bool OrbitCameraController::onKeyEvent(const KeyboardEvent& keyEvent)
{
	return false;
}

KRR_NAMESPACE_END

