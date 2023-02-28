#pragma once
#include "common.h"
#include "mesh.h"
#include "light.h"
#include "camera.h"
#include "texture.h"
#include "interop.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "render/lightsampler.h"

KRR_NAMESPACE_BEGIN
using namespace io;

class RTScene;
class VKScene;

class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene();
	~Scene() = default;

	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update();
	bool getChanges() const { return mHasChanges; };
	void renderUI();

	Camera& getCamera() { return *mpCamera; }
	CameraController& getCameraController() { return *mpCameraController; }

	void setCamera(const Camera &camera) { *mpCamera = camera; }
	void setCameraController(const OrbitCameraController &cameraController) {
		*mpCameraController = cameraController;
	}
	void addInfiniteLight(const InfiniteLight& infiniteLight);
	
	void loadConfig(const json &config) { 
		mpCamera = std::make_shared<Camera>(config.at("camera")); 
		mpCameraController = std::make_shared<OrbitCameraController>(config.at("cameraController"));
		mpCameraController->setCamera(mpCamera);
		if (config.contains("envLights")) 
			for (auto &light : config.at("envLights")) {
				InfiniteLight l{};
				from_json(light, l);
				addInfiniteLight(l);
			}
	}
	
	AABB getAABB() const { return mAABB; }

	friend void to_json(json& j, const Scene& scene) { 
		j = json{ 
			{ "camera", *scene.mpCamera }, 
			{ "cameraController", *std::dynamic_pointer_cast
				<OrbitCameraController>(scene.mpCameraController) },
		};
	}

	std::vector<Mesh> meshes;
	std::vector<Material> materials{};
	std::vector<InfiniteLight> infiniteLights{};

	Camera::SharedPtr mpCamera;
	OrbitCameraController::SharedPtr mpCameraController;
	AABB mAABB;
	bool mHasChanges = false;

	std::shared_ptr<RTScene> mpSceneRT;
	void initializeSceneRT();
};

namespace rt {
struct SceneData {
	inter::vector<Material> *materials{};
	inter::vector<MeshData> *meshes{};
	inter::vector<Light> *lights{};
	inter::vector<InfiniteLight> *infiniteLights{};
	LightSampler lightSampler;
};
}

class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene() = default;
	RTScene(Scene* scene) : mpScene(scene){}
	~RTScene() = default;

	void toDevice();
	void renderUI();
	void updateSceneData();
	rt::SceneData getSceneData() const { return mDeviceData; }

private:
	void processLights();

	Scene* mpScene;
	rt::SceneData mDeviceData;
};

class VKScene {
public:
	using SharedPtr = std::shared_ptr<VKScene>;
	
	VKScene() = default;
	VKScene(Scene* scene) : mpScene(scene) {}
	~VKScene() = default;

private:
	Scene* mpScene;
};

KRR_NAMESPACE_END