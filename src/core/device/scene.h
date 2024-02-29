#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include "common.h"
#include "mesh.h"
#include "light.h"
#include "camera.h"
#include "texture.h"

#include "device/gpustd.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "render/lightsampler.h"
#include "render/media.h"

NAMESPACE_BEGIN(krr)
class Scene;

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

struct OptixSceneParameters {
	bool enableAnimation{false};
	bool buildMultilevel{false};
	bool enableMotionBlur{false};
	float worldStartTime{0}, worldEndTime{1};

	friend void from_json(const json& j, OptixSceneParameters& p) {
		p.enableAnimation = j.value("animated", false);
		p.buildMultilevel = j.value("multilevel", false);
		p.enableMotionBlur = j.value("motionblur", false);
		p.worldStartTime = j.value("starttime", 0.f);
		p.worldEndTime = j.value("endtime", 1.f);
	}
};

class OptixScene;
class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene(std::shared_ptr<Scene> scene);
	~RTScene() = default;

	void update();
	void uploadSceneData(const OptixSceneParameters& parameters);
	void updateSceneData();

	rt::SceneData getSceneData() const;
	std::shared_ptr<Scene> getScene() const;
	std::shared_ptr<OptixScene> getOptixScene() const { return mOptixScene; }

	std::vector<rt::MaterialData> &getMaterialData() { return mMaterials; }
	std::vector<rt::MeshData> &getMeshData() { return mMeshes; }
	std::vector<rt::InstanceData> &getInstanceData() { return mInstances; }
	std::vector<rt::Light> &getLightData() { return mLights; }
	std::vector<Medium> &getMediumData() { return mMedium; }

	TypedBuffer<rt::MaterialData> &getMaterialBuffer() { return mMaterialsBuffer; }
	TypedBuffer<rt::MeshData> &getMeshBuffer() { return mMeshesBuffer; }
	TypedBuffer<rt::InstanceData> &getInstanceBuffer() { return mInstancesBuffer; }
	TypedBuffer<rt::Light> &getLightBuffer() { return mLightsBuffer; }
	TypedBuffer<Medium> &getMediumBuffer() { return mMediumBuffer; }

private:
	void uploadSceneMaterialData();
	void uploadSceneMeshData();
	void uploadSceneInstanceData();
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
	std::vector<NanoVDBMedium<float>> mNanoVDBMedium;
	TypedBuffer<NanoVDBMedium<float>> mNanoVDBMediumBuffer;
	std::vector<NanoVDBMedium<Array3f>> mNanoVDBRGBMedium;
	TypedBuffer<NanoVDBMedium<Array3f>> mNanoVDBRGBMediumBuffer;

	std::vector<Medium> mMedium;
	TypedBuffer<Medium> mMediumBuffer;

	UniformLightSampler mLightSampler;
	TypedBuffer<UniformLightSampler> mLightSamplerBuffer;

	std::weak_ptr<Scene> mScene;
	std::shared_ptr<OptixScene> mOptixScene;
};

template <typename Integrator> 
struct LaunchParameters {};

NAMESPACE_END(krr)