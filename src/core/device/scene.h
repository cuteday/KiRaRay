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
#include "device/container.h"
#include "render/lightsampler.h"
#include "render/media.h"

NAMESPACE_BEGIN(krr)
class Scene;
class SceneLight;

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
		p.enableAnimation  = j.value("animated", true);
		p.buildMultilevel  = j.value("multilevel", false);
		p.enableMotionBlur = j.value("motionblur", false);
		p.worldStartTime   = j.value("starttime", 0.f);
		p.worldEndTime	   = j.value("endtime", 1.f);
	}
};

template <typename T1, typename T2, typename Tp>
class SceneStorage;

template <typename CpuType, typename GpuType, typename ...GpuTypes> 
class SceneStorage<CpuType, GpuType, TypePack<GpuTypes...>> {
public:
	using Types = TypePack<GpuTypes...>;
	struct Record {
		char *base;
		size_t index;
		size_t size;
		unsigned int type;
	};
	SceneStorage() = default;
	virtual ~SceneStorage() = default;

	void clear() {
		mRecords.clear();
		mPointers.clear();
		mData.clear();
	}

	template <typename T> 
	gpu::vector<T> &getData() { 
		return *mData.template get<T>();
	}

	void addPointer(GpuType entity) {
		mPointers.emplace_back(entity);
	}

	void addPointers(const std::vector<std::shared_ptr<CpuType>> entities) {
		mPointers.reserve(mPointers.size() + entities.size());
		cudaDeviceSynchronize();
		for (auto entity : entities) {
			Record record		  = mRecords[entity];
			CUdeviceptr devicePtr =
				*reinterpret_cast<CUdeviceptr *>(record.base) + record.size * record.index;
			mPointers.emplace_back(reinterpret_cast<void *>(devicePtr), record.type);
		}
	}

	GpuType getPointer(std::weak_ptr<CpuType> entity) {
		Record record		  = mRecords[entity];
		CUdeviceptr devicePtr =
			*reinterpret_cast<CUdeviceptr *>(record.base) + record.size * record.index;
		return GpuType(reinterpret_cast<void *>(devicePtr), record.type);
	}

	gpu::vector<GpuType> &getPointers() { return mPointers; }

	template <typename T>
	void pushEntity(std::weak_ptr<CpuType> entity, const T& data) {
		auto ptr		 = mData.template get<T>();
		mRecords[entity] = {reinterpret_cast<char *>(ptr), ptr->size(), sizeof(T),
							IndexOf<T, Types>::count + 1};	// type 0 means null
		ptr->push_back(data);
	}

	template <typename T, typename ...Args>
	void emplaceEntity(std::weak_ptr<CpuType> entity, Args&&... args) {
		auto ptr		 = mData.template get<T>();
		mRecords[entity] = {reinterpret_cast<char *>(ptr), ptr->size(), sizeof(T),
							IndexOf<T, Types>::count + 1};
		ptr->emplace_back(std::forward<Args>(args)...);
	}

private:
	std::map<std::weak_ptr<CpuType>, Record, std::owner_less<std::weak_ptr<CpuType>>> mRecords;
	gpu::vector<GpuType> mPointers;
	gpu::multi_vector<Types> mData;
};

class SceneObject : public TaggedPointer<rt::PointLight, rt::DirectionalLight, rt::SpotLight, 
	rt::InfiniteLight, HomogeneousMedium, NanoVDBMedium<float>, NanoVDBMedium<Array3f>,
	rt::MaterialData, rt::InstanceData> {
public:
	SceneObject() = default;

	template <typename T> 
	SceneObject(T *ptr): TaggedPointer(ptr) {
		data = std::make_shared<Blob>(sizeof(T));
	}

	void getObjectData(SceneGraphLeaf::SharedPtr object, bool initialize = false) const {
		auto func = [&](auto ptr) -> void { ptr->getObjectData(object, data, initialize); };
		return dispatch(func);
	}

	std::shared_ptr<Blob> data;
};

class OptixScene;
class RTScene {
public:
	using SharedPtr = std::shared_ptr<RTScene>;

	RTScene(std::shared_ptr<Scene> scene);
	~RTScene() = default;

	void updateAccelStructure();
	void uploadSceneData(const OptixSceneParameters& parameters);
	void updateSceneData();
	void updateManagedObject(SceneGraphLeaf::SharedPtr object);
	void uploadManagedObject(SceneGraphLeaf::SharedPtr leaf, SceneObject object);

	rt::SceneData getSceneData();
	std::shared_ptr<Scene> getScene() const;
	std::shared_ptr<OptixScene> getOptixScene() const { return mOptixScene; }

	gpu::vector<rt::MaterialData> &getMaterialData() { return mMaterials; }
	gpu::vector<rt::MeshData> &getMeshData() { return mMeshes; }
	gpu::vector<rt::InstanceData> &getInstanceData() { return mInstances; }
	gpu::vector<rt::Light> &getLightData() { return mLightStorage.getPointers(); }
	gpu::vector<Medium> &getMediumData() { return mMediumStorage.getPointers(); }

private:
	void uploadSceneMaterialData();
	void uploadSceneMeshData();
	void uploadSceneInstanceData();
	void uploadSceneLightData();
	void uploadSceneMediumData();

	gpu::vector<rt::MaterialData> mMaterials;
	gpu::vector<rt::MeshData> mMeshes;
	gpu::vector<rt::InstanceData> mInstances;
	
	SceneStorage<Volume, Medium, Medium::Types> mMediumStorage;
	SceneStorage<SceneLight, rt::Light, rt::Light::Types> mLightStorage;

	std::map<std::weak_ptr<SceneGraphLeaf>, SceneObject,
			 std::owner_less<std::weak_ptr<SceneGraphLeaf>>> mManagedObjects;

	TypedBuffer<UniformLightSampler> mLightSamplerBuffer;

	std::weak_ptr<Scene> mScene;
	std::shared_ptr<OptixScene> mOptixScene;
};

template <typename Integrator> 
struct LaunchParameters {};

NAMESPACE_END(krr)