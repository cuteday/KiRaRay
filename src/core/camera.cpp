#include "camera.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

bool Camera::update()
{
	if (mPreserveHeight)
		mData.frameSize.x = mData.aspectRatio * mData.frameSize.y;
	else mData.frameSize.y = mData.frameSize.x / mData.aspectRatio;

	float fov = atan2(mData.frameSize.y * 0.5, mData.focalLength); // vertical fov
	
	mData.w = mData.focalLength * normalize(mData.target - mData.pos);
	mData.u = normalize(cross(mData.w, mData.up)) * mData.frameSize.x * 0.5f;
	mData.v = normalize(cross(mData.u, mData.w)) * mData.frameSize.y * 0.5f;

	// update jitter value
	mData.jitter = mpJitterSampler.get2D() - vec2f(1);

	// checking if camera data have changes
	bool hasChanges = false;
	mDataPrev = mData;
	return hasChanges;
}

void Camera::renderUI()
{
	ui::Text("Position: %f, %f, %f", mData.pos.x, mData.pos.y, mData.pos.z);
	ui::Text("Target: %f, %f, %f", mData.target.x, mData.target.y, mData.target.z);
	ui::Text("Focal vector: %f, %f, %f", mData.w.x, mData.w.y, mData.w.z);
	ui::Text("Sensor right: %f, %f, %f", mData.u.x, mData.u.y, mData.u.z);
	ui::Text("Sensor up: %f, %f, %f", mData.v.x, mData.v.y, mData.v.z);
}

bool OrbitCameraController::update()
{	
	quat rotate = normalize(quat(mData.yaw, mData.pitch, 0));
	vec3f forward = rotate * vec3f(0, 0, -1);
	vec3f pos = mData.target - forward * mData.radius;

	mpCamera->setPosition(pos);
	mpCamera->setTarget(mData.target);

	bool hasChanges = false;
	hasChanges |= (bool)memcmp(&mData, &mDataPrev, sizeof(CameraControllerData));
	mDataPrev = mData;
	return hasChanges;
}

bool OrbitCameraController::onMouseEvent(const MouseEvent& mouseEvent)
{

	switch (mouseEvent.type) {
	case io::MouseEvent::Type::Wheel:
		mData.radius -= mouseEvent.wheelDelta.y * mZoomSpeed;
		mData.radius = clamp(mData.radius, 0.1f, 1e5f);
		return true;
	case io::MouseEvent::Type::LeftButtonDown:
		mOrbiting = true;
		return true;
	case io::MouseEvent::Type::LeftButtonUp:
		mOrbiting = false;
		return true;
	case io::MouseEvent::Type::Move:
		vec2f curMousePos = mouseEvent.pos;
		vec2f deltaPos = curMousePos - mLastMousePos;
		if (mPanning && mOrbiting) {
			mData.target -= mpCamera->getRight() * mData.radius * mPanSpeed * deltaPos.x;
			mData.target += mpCamera->getUp() * mData.radius * mPanSpeed * deltaPos.y;
		}
		else if(mOrbiting){	
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
	if (keyEvent.key == KeyboardEvent::Key::LeftShift) {
		if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
			mPanning = true;
		else if (keyEvent.type == KeyboardEvent::Type::KeyReleased)
			mPanning = false;
		return true;
	}
	return false;
}

KRR_NAMESPACE_END

