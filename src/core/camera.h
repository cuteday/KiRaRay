#pragma once
#include "raytracing.h"
#include "scenenode.h"
#include "render/sampling.h"
#include "sampler.h"
#include "input.h"
#include "krrmath/functors.h"

KRR_NAMESPACE_BEGIN
using namespace io;
class Scene;

namespace rt {
struct CameraData {
	Vector2f filmSize{42.666667f, 24.0f}; // sensor size in mm [width, height]
	float focalLength{21};				  // distance from sensor to lens, in mm
	float focalDistance{10};			  // distance from lens to focal point, in scene units (m)
	float lensRadius{0};				  // aperture radius, in mm
	float aspectRatio{1.777777f};		  // width divides height
	float shutterOpen{0};				  // shutter open time
	float shutterTime{0};				  // shutter time (default 0: disable motion blur)

	Transformation transform;
	Medium medium{nullptr}; // the ray is inside the medium

	KRR_CALLABLE Ray getRay(Vector2i pixel, Vector2i frameSize, Sampler &sampler) {
		Ray ray{};
		/* 1. Statified sample on the film plane (within the fragment) */
		Vector2f p	 = (Vector2f) pixel + Vector2f(0.5f) + sampler.get2D();
		Vector2f ndc = Vector2f(2 * p) / Vector2f(frameSize) + Vector2f(-1.f); // ndc in [-1, 1]^2
		float fov	 = atan2(filmSize[1] * 0.5f, focalLength);

		ray.medium = medium;
		ray.time   = shutterOpen + shutterTime * sampler.get1D();
		Vector3f focalDirection =
			Vector3f{tan(fov) * aspectRatio * ndc[0], tan(fov) * ndc[1], -1}.normalized();

		if (lensRadius > M_EPSILON) {										   /*Thin lens*/
			/* 2. Sample the lens (uniform) */
			Vector2f apertureSample = uniformSampleDisk(sampler.get2D());
			ray.origin.head<2>()	= lensRadius * apertureSample;
			ray.dir = (focalDirection * focalDistance - ray.origin).normalized();
		} else { /*Pin hole*/
			ray.origin = Vector3f::Zero();
			ray.dir	   = focalDirection;
		}
		return transform(ray);
	}

	friend void to_json(json& j, const CameraData& camera) {
		j = json{{"focalLength", camera.focalLength},
				 {"focalDistance", camera.focalDistance},
				 {"lensRadius", camera.lensRadius},
				 {"aspectRatio", camera.aspectRatio},
				 {"shutterOpen", camera.shutterOpen},
				 {"shutterTime", camera.shutterTime}};
	}

	friend void from_json(const json& j, CameraData& camera) {
		/* only does incremental update */
		camera.focalLength = j.value("focalLength", camera.focalLength);
		camera.focalDistance = j.value("focalDistance", camera.focalDistance);
		camera.lensRadius = j.value("lensRadius", camera.lensRadius);
		camera.aspectRatio = j.value("aspectRatio", camera.aspectRatio);
		camera.shutterOpen = j.value("shutterOpen", camera.shutterOpen);
		camera.shutterTime	 = j.value("shutterTime", camera.shutterTime);
	}
};
} // namespace rt

class Camera : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<Camera>;
	using CameraData = rt::CameraData;
	Camera()		= default;
	Camera(std::weak_ptr<Scene> scene, const CameraData& data):
		mScene(scene), mData(data) {}

	bool update();
	void renderUI() override;

	ContentFlags getContentFlags() const override { return ContentFlags::Camera; }
	std::shared_ptr<SceneGraphLeaf> clone() override;

	float getAspectRatio() const { return mData.aspectRatio; }
	Vector3f getPosition() const { return getTransform().translation(); }
	Vector3f getTarget() const { return getPosition() + getForward() * getFocalDistance(); }
	Vector3f getForward() const { return getRotation() * -Vector3f::UnitZ();; }
	Vector3f getUp() const { return getRotation() * Vector3f::UnitY(); }
	Vector3f getRight() const { return getRotation() * Vector3f::UnitX(); }
	Vector2f getFilmSize() const { return mData.filmSize; }
	Affine3f getTransform() const { return getNode()->getGlobalTransform(); }
	Matrix3f getRotation() const { return getTransform().rotation(); }
	float getLensRadius() const { return mData.lensRadius; }
	float getFocalDistance() const { return mData.focalDistance; }
	float getFocalLength() const { return mData.focalLength; }
	float getShutterOpen() const { return mData.shutterOpen; }
	float getShutterTime() const { return mData.shutterTime; }
	const rt::CameraData& getCameraData() const { return mData; }

	void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	void setFilmSize(Vector2f& size) { mData.filmSize = size; }
	void setFocalDistance(float focalDistance) { mData.focalDistance = focalDistance; }
	void setFocalLength(float focalLength) { mData.focalLength = focalLength; }
	void setLensRadius(float lensRadius) { mData.lensRadius = lensRadius; }
	void setShutterOpen(float shutterOpen) { mData.shutterOpen = shutterOpen; }
	void setShutterTime(float shutterTime) { mData.shutterTime = shutterTime; }
	void setChanged() { mHasChanges = true; }
	void setScene(std::weak_ptr<Scene> scene) { mScene = scene; }

	Matrix4f getViewMatrix() const;
	Matrix4f getProjectionMatrix() const;
	Matrix4f getViewProjectionMatrix() const;

protected:
	KRR_CLASS_DEFINE(Camera, mData);
	std::weak_ptr<Scene> mScene;
	rt::CameraData mData, mDataPrev;
	bool mHasChanges	 = false;
	bool mPreserveHeight = true;	// preserve sensor height on aspect ratio changes.
};

class CameraController{
public:
	using SharedPtr = std::shared_ptr<CameraController>;

	virtual bool update() = 0;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) = 0;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) = 0;
	virtual void renderUI() {}
	virtual void setCamera(const Camera::SharedPtr &pCamera) { mCamera = pCamera; }
		
protected:
	CameraController() = default;
	CameraController(const Camera::SharedPtr& pCamera){ 
		setCamera(pCamera);
	}

	Camera::SharedPtr mCamera{};
	float mSpeed = 1.f;
};

class OrbitCameraController: public CameraController{
public:
	using SharedPtr = std::shared_ptr<OrbitCameraController>;
	struct CameraControllerData{
		Vector3f target{0, 0, 0};
		float radius = 5;
		float pitch	 = 0;
		float yaw	 = 0;
		
		KRR_CLASS_DEFINE(CameraControllerData, radius, pitch, yaw, target);
	};

	OrbitCameraController() = default;
	OrbitCameraController(Camera::SharedPtr pCamera): CameraController(pCamera) {}
	
	virtual bool update() override;
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;
	virtual void renderUI() override;
	virtual void setCamera(const Camera::SharedPtr &pCamera) override;

private:
	KRR_CLASS_DEFINE(OrbitCameraController, mData);
	CameraControllerData mData, mDataPrev;
	Vector2f mLastMousePos;
	float mDampling	  = 1;
	bool mOrbiting	  = false;
	bool mPanning	  = false;
	float mOrbitSpeed = 1;
	float mPanSpeed	  = 1;
	float mZoomSpeed  = 0.1;
};

KRR_NAMESPACE_END