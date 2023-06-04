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
	std::vector<MeshInstance::SharedPtr> &getMeshInstances() { return mGraph->getMeshInstances(); }
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
	bool mHasChanges	  = false;
	bool mEnableAnimation = false;

	std::shared_ptr<RTScene> mSceneRT;
	std::shared_ptr<VKScene> mSceneVK;
	void initializeSceneRT();
	void initializeSceneVK(nvrhi::vulkan::IDevice* device,
		std::shared_ptr<DescriptorTableManager> descriptorTable = nullptr);
	std::shared_ptr<RTScene> getSceneRT() const { return mSceneRT; }
	std::shared_ptr<VKScene> getSceneVK() const { return mSceneVK; }
};

namespace rt {

class SceneData {
public:
	TypedBufferView<MaterialData> materials{};
	TypedBufferView<MeshData> meshes{};
	TypedBufferView<InstanceData> instances{};
	TypedBufferView<Light> lights{};
	TypedBufferView<InfiniteLight> infiniteLights{};
	LightSampler lightSampler;
};
}

class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene() = default;
	RTScene(Scene::SharedPtr scene) : mScene(scene){}
	~RTScene() = default;

	void toDevice();
	void uploadSceneData();
	void updateSceneData();
	rt::SceneData getSceneData() const;

	std::vector<rt::MaterialData> &getMaterialData() { return mMaterials; }
	std::vector<rt::MeshData> &getMeshData() { return mMeshes; }
	std::vector<rt::InstanceData> &getInstanceData() { return mInstances; }
	std::vector<Light> &getLightData() { return mLights; }
	std::vector<InfiniteLight> &getInfiniteLightData() { return mInfiniteLights; }

	TypedBuffer<rt::MaterialData> &getMaterialBuffer() { return mMaterialsBuffer; }
	TypedBuffer<rt::MeshData> &getMeshBuffer() { return mMeshesBuffer; }
	TypedBuffer<rt::InstanceData> &getInstanceBuffer() { return mInstancesBuffer; }
	TypedBuffer<Light> &getLightBuffer() { return mLightsBuffer; }
	TypedBuffer<InfiniteLight> &getInfiniteLightBuffer() { return mInfiniteLightsBuffer; }

private:
	void processLights();

	std::vector<rt::MaterialData> mMaterials;
	TypedBuffer<rt::MaterialData> mMaterialsBuffer;
	std::vector<rt::MeshData> mMeshes;
	TypedBuffer<rt::MeshData> mMeshesBuffer;
	std::vector<rt::InstanceData> mInstances;
	TypedBuffer<rt::InstanceData> mInstancesBuffer;
	std::vector<Light> mLights;
	TypedBuffer<Light> mLightsBuffer;
	std::vector<InfiniteLight> mInfiniteLights; 
	TypedBuffer<InfiniteLight> mInfiniteLightsBuffer;
	UniformLightSampler mLightSampler;
	TypedBuffer<UniformLightSampler> mLightSamplerBuffer;

	std::weak_ptr<Scene> mScene;
};


KRR_NAMESPACE_END