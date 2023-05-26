#pragma once
#include "common.h"
#include "mesh.h"
#include "light.h"
#include "camera.h"
#include "texture.h"
#include "scenegraph.h"
#include "interop.h"

#include "device/buffer.h"
#include "device/memory.h"
#include "render/lightsampler.h"

#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

using namespace io;

class RTScene;
class VKScene;
class DescriptorTableManager;

class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene();
	~Scene() = default;

	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update(size_t frameIndex);
	bool getChanges() const { return mHasChanges; }
	void renderUI();

	Camera::SharedPtr getCamera() { return mCamera; }
	CameraController::SharedPtr getCameraController() { return mCameraController; }
	SceneGraph::SharedPtr getSceneGraph() { return mGraph; }

	std::vector<MeshInstance::SharedPtr> &getMeshInstances() { return mGraph->getMeshInstances(); }
	std::vector<Mesh::SharedPtr> &getMeshes() { return mGraph->getMeshes(); }
	std::vector<Material::SharedPtr> &getMaterials() { return mGraph->getMaterials(); }
	void addMesh(Mesh::SharedPtr mesh) { mGraph->addMesh(mesh); }
	void addMaterial(Material::SharedPtr material) { mGraph->addMaterial(material); }

	void setCamera(Camera::SharedPtr camera) { mCamera = camera; }
	void setCameraController(OrbitCameraController::SharedPtr cameraController) {
		mCameraController = cameraController;
	}
	void addEnvironmentMap(Texture::SharedPtr infiniteLight);
	
	void loadConfig(const json &config) { 
		mCamera = std::make_shared<Camera>(config.at("camera")); 
		mCameraController = std::make_shared<OrbitCameraController>(config.at("cameraController"));
		mCameraController->setCamera(mCamera);
	}
	
	AABB getBoundingBox() const { return mGraph->getRoot()->getGlobalBoundingBox(); }

	friend void to_json(json& j, const Scene& scene) { 
		j = json{ 
			{ "camera", *scene.mCamera }, 
			{ "cameraController", *std::dynamic_pointer_cast
				<OrbitCameraController>(scene.mCameraController) },
		};
	}

	SceneGraph::SharedPtr mGraph;
	Camera::SharedPtr mCamera;
	OrbitCameraController::SharedPtr mCameraController;
	std::vector<Texture::SharedPtr> environments;
	bool mHasChanges = false;

	std::shared_ptr<RTScene> mSceneRT;
	std::shared_ptr<VKScene> mSceneVK;
	void initializeSceneRT();
	void initializeSceneVK(nvrhi::vulkan::IDevice* device,
		std::shared_ptr<DescriptorTableManager> descriptorTable = nullptr);
};

namespace rt {

class SceneData {
public:
	inter::vector<MaterialData> *materials{};
	inter::vector<MeshData> *meshes{};
	inter::vector<InstanceData> *instances{};
	inter::vector<Light> *lights{};
	inter::vector<InfiniteLight> *infiniteLights{};
	LightSampler lightSampler;
};
}

class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene() = default;
	RTScene(Scene* scene) : mScene(scene){}
	~RTScene() = default;

	void toDevice();
	void renderUI();
	void updateSceneData();
	const rt::SceneData &getSceneData() const { return mDeviceData; }

private:
	void processLights();

	Scene* mScene;
	rt::SceneData mDeviceData;
};


KRR_NAMESPACE_END