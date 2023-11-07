#pragma once
#include "common.h"
#include "mesh.h"
#include "light.h"
#include "camera.h"
#include "texture.h"
#include "scenegraph.h"
#include "device/gpustd.h"

#include "device/buffer.h"
#include "device/memory.h"
#include "render/lightsampler.h"
#include "render/media.h"

#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

using namespace io;

class RTScene;
class VKScene;
class DescriptorTableManager;

class Scene : public std::enable_shared_from_this<Scene> {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene();
	~Scene() = default;

	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update(size_t frameIndex, double currentTime);
	bool getChanges() const { return mHasChanges; }
	void renderUI();

	Camera::SharedPtr getCamera() { return mCamera; }
	CameraController::SharedPtr getCameraController() { return mCameraController; }
	SceneGraph::SharedPtr getSceneGraph() { return mGraph; }

	std::vector<Mesh::SharedPtr> &getMeshes() { return mGraph->getMeshes(); }
	std::vector<Material::SharedPtr> &getMaterials() { return mGraph->getMaterials(); }
	std::vector<SceneAnimation::SharedPtr> &getAnimations() { return mGraph->getAnimations(); }
	std::vector<SceneLight::SharedPtr> &getLights() { return mGraph->getLights(); }
	std::vector<MeshInstance::SharedPtr> &getMeshInstances() { return mGraph->getMeshInstances(); }
	std::vector<Volume::SharedPtr> &getMedia() { return mGraph->getMedia(); }

	void addMesh(Mesh::SharedPtr mesh) { mGraph->addMesh(mesh); }
	void addMaterial(Material::SharedPtr material) { mGraph->addMaterial(material); }

	void setCamera(Camera::SharedPtr camera) { mCamera = camera; }
	void setCameraController(OrbitCameraController::SharedPtr cameraController) {
		mCameraController = cameraController;
	}

	json getConfig() const { return mConfig; }
	void setConfig(const json &config, bool update = true);
	AABB getBoundingBox() const { return mGraph->getRoot()->getGlobalBoundingBox(); }

	friend void to_json(json& j, const Scene& scene) { 
		j = scene.getConfig();
		j.update(json{ 
			{ "camera", *scene.mCamera }, 
			{ "cameraController", *std::dynamic_pointer_cast
				<OrbitCameraController>(scene.mCameraController) },
		});
	}

	json mConfig;
	SceneGraph::SharedPtr mGraph;
	Camera::SharedPtr mCamera;
	OrbitCameraController::SharedPtr mCameraController;
	bool mHasChanges	  = false;
	bool mEnableAnimation = true;

	std::shared_ptr<RTScene> mSceneRT;
	std::shared_ptr<VKScene> mSceneVK;
	void initializeSceneRT();
	void initializeSceneVK(nvrhi::vulkan::IDevice* device,
		std::shared_ptr<DescriptorTableManager> descriptorTable = nullptr);
	std::shared_ptr<RTScene> getSceneRT() const { return mSceneRT; }
	std::shared_ptr<VKScene> getSceneVK() const { return mSceneVK; }
};

KRR_NAMESPACE_END