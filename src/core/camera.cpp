#include "camera.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

bool Camera::update(){
	if (mScene.lock()->getSceneRT()) {
		/* Ray-tracing enabled, update medium info */
		for (auto medium : mScene.lock()->getMedia()) {
			if (medium->getNode()->getGlobalBoundingBox().contains(getPosition())) {
				mData.medium =
					mScene.lock()->getSceneRT()->getMediumData()[medium->getMediumId()];
			}
		}
	}
	/* Update parameters in data. */
	if (mPreserveHeight) mData.filmSize[0] = mData.aspectRatio * mData.filmSize[1];
	else mData.filmSize[1] = mData.filmSize[0] / mData.aspectRatio;

	float fovY = atan2(mData.filmSize[1] * 0.5, mData.focalLength);
	mData.w = normalize(mData.target - mData.pos) * mData.focalDistance;
	mData.u = normalize(cross(mData.w, mData.up)) * tan(fovY) * mData.focalDistance * mData.aspectRatio;
	mData.v = normalize(cross(mData.u, mData.w)) * tan(fovY) * mData.focalDistance;
	bool hasChanges = (bool)memcmp(&mData, &mDataPrev, sizeof(rt::CameraData));;
	mDataPrev = mData;
	return hasChanges;
}

void Camera::renderUI() {
	ui::DragFloat("Lens radius", &mData.lensRadius, 0.001f, 0.f, 100.f);
	ui::DragFloat("Focal length", &mData.focalLength, 0.01f, 1.f, 1000.f);
	ui::DragFloat("Focal distance", &mData.focalDistance, 0.01f, 1.f, 100.f);
}

Matrix4f Camera::getViewMatrix() const {
	return look_at(mData.pos, mData.target, Vector3f{0, 1, 0});
}

Matrix4f Camera::getProjectionMatrix() const {
	float fovy = 2 * atan2(mData.filmSize[1] * 0.5f, mData.focalLength);
	return perspective(fovy, mData.aspectRatio, 0.01f, 1000.f);
}

Matrix4f Camera::getViewProjectionMatrix() const {
	return getProjectionMatrix() * getViewMatrix();
}

bool OrbitCameraController::update(){
	Quaternionf rotate = Quaternionf::fromEuler(mData.yaw, mData.pitch, 0);
	rotate.normalize();
	Vector3f forward = rotate * Vector3f(0, 0, -1);
	Vector3f pos = mData.target - forward * mData.radius;

	mCamera->setPosition(pos);
	mCamera->setTarget(mData.target);

	bool hasChanges = (bool)memcmp(&mData, &mDataPrev, sizeof(CameraControllerData));
	mDataPrev = mData;
	return hasChanges;
}

bool OrbitCameraController::onMouseEvent(const MouseEvent& mouseEvent){
	switch (mouseEvent.type) {
	case io::MouseEvent::Type::Wheel:
		mData.radius -= mouseEvent.wheelDelta[1] * clamp(mZoomSpeed * mData.radius, 1e-2f, 1e1f);
		mData.radius = clamp(mData.radius, 0.1f, 1e5f);
		return true;
	case io::MouseEvent::Type::LeftButtonDown:
		mOrbiting = true;
		return true;
	case io::MouseEvent::Type::LeftButtonUp:
		mOrbiting = false;
		return true;
	case io::MouseEvent::Type::MiddleButtonDown:
		mPanning = mOrbiting = true;
		return true;
	case io::MouseEvent::Type::MiddleButtonUp:
		mPanning = mOrbiting = false;
		return true;
	case io::MouseEvent::Type::Move:
		Vector2f curMousePos = mouseEvent.pos;
		Vector2f deltaPos = curMousePos - mLastMousePos;
		mLastMousePos = curMousePos;
		if (mPanning && mOrbiting) {
			mData.target -= mCamera->getRight() * mData.radius * mPanSpeed * deltaPos[0];
			mData.target += mCamera->getUp() * mData.radius * mPanSpeed * deltaPos[1];
		}
		else if(mOrbiting){	
			mData.yaw -= deltaPos[0] * mOrbitSpeed;
			mData.pitch -= deltaPos[1] * mOrbitSpeed;
			mData.yaw = fmod(mData.yaw, 2 * M_PI);
			mData.pitch = clamp(mData.pitch, -M_PI / 2, M_PI / 2);	
		}
		return mOrbiting;
	}
	return false;
}

bool OrbitCameraController::onKeyEvent(const KeyboardEvent& keyEvent){
	if (keyEvent.key == KeyboardEvent::Key::LeftShift) {
		if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
			mPanning = true;
		else if (keyEvent.type == KeyboardEvent::Type::KeyReleased)
			mPanning = false;
		return true;
	}
	return false;
}

void OrbitCameraController::renderUI() {
	ui::DragFloat("Orbit radius", &mData.radius, 1e-2, 0.1f, 1e5f);
	ui::DragFloat("Yaw", &mData.yaw, 1e-2, 0, 2 * M_PI);
	ui::DragFloat("Pitch", &mData.pitch, 1e-2, -M_PI / 2, M_PI / 2);
	ui::DragFloat3("Target", (float*)&mData.target, 1e-1, -1e5f, 1e5f);
}

void OrbitCameraController::setCamera(const Camera::SharedPtr &pCamera) { 
	CameraController::setCamera(pCamera);
	update();
}

KRR_NAMESPACE_END