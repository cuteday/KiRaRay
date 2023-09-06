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
	void addEnvironmentMap(Texture::SharedPtr infiniteLight);
	void loadConfig(const json &config);
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
	bool mEnableAnimation = true;

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
	TypedBufferView<rt::MaterialData> materials{};
	TypedBufferView<rt::MeshData> meshes{};
	TypedBufferView<rt::InstanceData> instances{};
	TypedBufferView<rt::Light> lights{};
	TypedBufferView<rt::InfiniteLight> infiniteLights{};
	LightSampler lightSampler;
};
}

class OptixScene;
class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene(Scene::SharedPtr scene);
	~RTScene() = default;

	void update();
	void uploadSceneData();
	void updateSceneData();

	rt::SceneData getSceneData() const;
	Scene::SharedPtr getScene() const { return mScene.lock(); }
	std::shared_ptr<OptixScene> getOptixScene() const { return mOptixScene; }

	std::vector<rt::MaterialData> &getMaterialData() { return mMaterials; }
	std::vector<rt::MeshData> &getMeshData() { return mMeshes; }
	std::vector<rt::InstanceData> &getInstanceData() { return mInstances; }
	std::vector<rt::Light> &getLightData() { return mLights; }
	std::vector<Medium> &getMediumData() { return mMedium; }
	std::vector<rt::InfiniteLight> &getInfiniteLightData() { return mInfiniteLights; }

	TypedBuffer<rt::MaterialData> &getMaterialBuffer() { return mMaterialsBuffer; }
	TypedBuffer<rt::MeshData> &getMeshBuffer() { return mMeshesBuffer; }
	TypedBuffer<rt::InstanceData> &getInstanceBuffer() { return mInstancesBuffer; }
	TypedBuffer<rt::Light> &getLightBuffer() { return mLightsBuffer; }
	TypedBuffer<Medium> &getMediumBuffer() { return mMediumBuffer; }
	TypedBuffer<rt::InfiniteLight> &getInfiniteLightBuffer() { return mInfiniteLightsBuffer; }

private:
	void uploadSceneLightData();
	void uploadSceneMediumData();

	std::vector<rt::MaterialData> mMaterials;
	TypedBuffer<rt::MaterialData> mMaterialsBuffer;
	std::vector<rt::MeshData> mMeshes;
	TypedBuffer<rt::MeshData> mMeshesBuffer;
	std::vector<rt::InstanceData> mInstances;
	TypedBuffer<rt::InstanceData> mInstancesBuffer;
	std::vector<rt::Light> mLights;
	TypedBuffer<rt::Light> mLightsBuffer;
	std::vector<rt::InfiniteLight> mInfiniteLights; 
	TypedBuffer<rt::InfiniteLight> mInfiniteLightsBuffer;

	std::vector<HomogeneousMedium> mHomogeneousMedium;
	TypedBuffer<HomogeneousMedium> mHomogeneousMediumBuffer;
	std::vector<Medium> mMedium;
	TypedBuffer<Medium> mMediumBuffer;

	UniformLightSampler mLightSampler;
	TypedBuffer<UniformLightSampler> mLightSamplerBuffer;

	std::weak_ptr<Scene> mScene;
	std::shared_ptr<OptixScene> mOptixScene;
};

KRR_NAMESPACE_END