#pragma once
#include "raytracing.h"
#include "render/sampling.h"
#include "sampler.h"
#include "input.h"

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
	float shutterTime{0};				  // shutter time (default disable motion blur)

	Vector3f pos{0, 0, 0};
	Vector3f target{0, 0, -1};
	Vector3f up{0, 1, 0};

	Vector3f u{1, 0, 0};  // camera right		[dependent to aspect ratio]
	Vector3f v{0, 1, 0};  // camera up			[dependent to aspect ratio]
	Vector3f w{0, 0, -1}; // camera forward

	Medium medium{nullptr}; // the ray is inside the medium

	KRR_CALLABLE Ray getRay(Vector2i pixel, Vector2i frameSize, Sampler &sampler) {
		Ray ray;
		/* 1. Statified sample on the film plane (within the fragment) */
		Vector2f p =
			(Vector2f) pixel + Vector2f(0.5f) + sampler.get2D(); // uniform sample + box filter
		Vector2f ndc = Vector2f(2 * p) / Vector2f(frameSize) + Vector2f(-1.f); // ndc in [-1, 1]^2
		if (lensRadius > M_EPSILON) {										   /*Thin lens*/
			/* 2. Sample the lens (uniform) */
			Vector3f focalPoint		= pos + ndc[0] * u + ndc[1] * v + w;
			Vector2f apertureSample = uniformSampleDisk(sampler.get2D());
			ray.origin				= pos + lensRadius * (apertureSample[0] * u.normalized() +
											  apertureSample[1] * v.normalized());
			ray.dir					= normalize(focalPoint - ray.origin);
		} else { /*Pin hole*/
			ray.origin = pos;
			ray.dir	   = normalize(ndc[0] * u + ndc[1] * v + w);
		}
		ray.medium = medium;
		ray.time   = shutterOpen + shutterTime * sampler.get1D();
		return ray;
	}

	friend void to_json(json& j, const CameraData& camera) {
		j = json{{"pos", camera.pos},
				 {"target", camera.target},
				 {"up", camera.up},
				 {"focalLength", camera.focalLength},
				 {"focalDistance", camera.focalDistance},
				 {"lensRadius", camera.lensRadius},
				 {"aspectRatio", camera.aspectRatio},
				 {"shutterOpen", camera.shutterOpen},
				 {"shutterTime", camera.shutterTime}};
	}

	friend void from_json(const json& j, CameraData& camera) {
		camera.pos = j.value("pos", Vector3f{0, 0, 0});
		camera.target = j.value("target", Vector3f{0, 0, -1});
		camera.up = j.value("up", Vector3f{0, 1, 0});
		camera.focalLength = j.value("focalLength", 21.f);
		camera.focalDistance = j.value("focalDistance", 10.f);
		camera.lensRadius = j.value("lensRadius", 0.f);
		camera.aspectRatio = j.value("aspectRatio", 1.777777f);
		camera.shutterOpen = j.value("shutterOpen", 0.f);
		camera.shutterTime = j.value("shutterTime", 0.f);
	}
};
} // namespace rt

class Camera {
public:
	using SharedPtr = std::shared_ptr<Camera>;
	Camera() = default;

	bool update();
	void renderUI();

	float getAspectRatio() const { return mData.aspectRatio; }
	Vector3f getPosition() const { return mData.pos; }
	Vector3f getTarget() const { return mData.target; }
	Vector3f getForward() const { return normalize(mData.target - mData.pos); }
	Vector3f getUp() const { return normalize(mData.v); }
	Vector3f getRight() const { return normalize(mData.u); }
	Vector2f getFilmSize() const { return mData.filmSize; }
	float getFocalDistance() const { return mData.focalDistance; }
	float getFocalLength() const { return mData.focalLength; }
	float getShutterOpen() const { return mData.shutterOpen; }
	float getShutterTime() const { return mData.shutterTime; }
	const rt::CameraData& getCameraData() const { return mData; }

	void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; }
	void setFilmSize(Vector2f& size) { mData.filmSize = size; }
	void setFocalDistance(float focalDistance) { mData.focalDistance = focalDistance; }
	void setFocalLength(float focalLength) { mData.focalLength = focalLength; }
	void setPosition(Vector3f& pos) { mData.pos = pos; }
	void setTarget(Vector3f& target) { mData.target = target; }
	void setUp(Vector3f& up) { mData.up = up; }
	void setShutterOpen(float shutterOpen) { mData.shutterOpen = shutterOpen; }
	void setShutterTime(float shutterTime) { mData.shutterTime = shutterTime; }
	void setScene(std::weak_ptr<Scene> scene) { mScene = scene; }

	Matrix4f getViewMatrix() const;
	Matrix4f getProjectionMatrix() const;
	Matrix4f getViewProjectionMatrix() const;

protected:
	KRR_CLASS_DEFINE(Camera, mData);
	std::weak_ptr<Scene> mScene;
	rt::CameraData mData, mDataPrev;
	bool mHasChanges = false;
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
		Vector3f up{0, 1, 0};
		float radius = 5;
		float pitch	 = 0;
		float yaw	 = 0;
		
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
	float mDampling	  = 1;
	bool mOrbiting	  = false;
	bool mPanning	  = false;
	float mOrbitSpeed = 1;
	float mPanSpeed	  = 1;
	float mZoomSpeed  = 0.1;
};

KRR_NAMESPACE_END