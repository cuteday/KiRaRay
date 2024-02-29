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
	TypedBufferView<MaterialData> materials{};
	TypedBufferView<MeshData> meshes{};
	TypedBufferView<InstanceData> instances{};
	TypedBufferView<Light> lights{};
	TypedBufferView<InfiniteLight> infiniteLights{};
	LightSampler lightSampler;
};
}

struct OptixSceneParameters {
	bool enableAnimation{true};
	bool buildMultilevel{false};
	bool enableMotionBlur{false};
	float worldStartTime{0}, worldEndTime{1};

	friend void from_json(const json& j, OptixSceneParameters& p) {
		p.enableAnimation = j.value("animated", true);
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

	rt::SceneData getSceneData();
	std::shared_ptr<Scene> getScene() const;
	std::shared_ptr<OptixScene> getOptixScene() const { return mOptixScene; }

	gpu::vector<rt::MaterialData> &getMaterialData() { return mMaterials; }
	gpu::vector<rt::MeshData> &getMeshData() { return mMeshes; }
	gpu::vector<rt::InstanceData> &getInstanceData() { return mInstances; }
	gpu::vector<rt::Light> &getLightData() { return mLights; }
	gpu::vector<Medium> &getMediumData() { return mMedium; }

private:
	void uploadSceneMaterialData();
	void uploadSceneMeshData();
	void uploadSceneInstanceData();
	void uploadSceneLightData();
	void uploadSceneMediumData();

	gpu::vector<rt::MaterialData> mMaterials;
	gpu::vector<rt::MeshData> mMeshes;
	gpu::vector<rt::InstanceData> mInstances;
	gpu::vector<rt::Light> mLights;
	gpu::vector<rt::InfiniteLight> mInfiniteLights;

	gpu::vector<HomogeneousMedium> mHomogeneousMedium;
	gpu::vector<NanoVDBMedium<float>> mNanoVDBMedium;
	gpu::vector<NanoVDBMedium<Array3f>> mNanoVDBRGBMedium;
	gpu::vector<Medium> mMedium;

	UniformLightSampler mLightSampler;
	TypedBuffer<UniformLightSampler> mLightSamplerBuffer;

	std::weak_ptr<Scene> mScene;
	std::shared_ptr<OptixScene> mOptixScene;
};

template <typename Integrator> 
struct LaunchParameters {};

NAMESPACE_END(krr)