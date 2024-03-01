#include "device/scene.h"
#include "device/optix.h"
#include "render/profiler/profiler.h"
#include "../scene.h"
#include "util/volume.h"

NAMESPACE_BEGIN(krr)

RTScene::RTScene(Scene::SharedPtr scene) : mScene(scene) {}

std::shared_ptr<Scene> RTScene::getScene() const { return mScene.lock(); }

void RTScene::uploadSceneData(const OptixSceneParameters &params) {
	// The order of upload is unchangeable since some of them are dependent to others.
	uploadSceneMaterialData();
	uploadSceneMediumData();
	uploadSceneMeshData();
	uploadSceneInstanceData();
	uploadSceneLightData();

	CUDA_SYNC_CHECK();
	if (params.buildMultilevel) 
		mOptixScene = std::make_shared<OptixSceneMultiLevel>(mScene.lock(), params);
	else mOptixScene = std::make_shared<OptixSceneSingleLevel>(mScene.lock(), params);
}

void RTScene::uploadSceneMeshData() {
	/* Upload mesh data to device... */
	auto &meshes = mScene.lock()->getMeshes();
	mMeshes.resize(meshes.size());
	for (size_t idx = 0; idx < meshes.size(); idx++) {
		const auto &mesh = meshes[idx];
		mMeshes[idx].positions.alloc_and_copy_from_host(mesh->positions);
		mMeshes[idx].normals.alloc_and_copy_from_host(mesh->normals);
		mMeshes[idx].texcoords.alloc_and_copy_from_host(mesh->texcoords);
		mMeshes[idx].tangents.alloc_and_copy_from_host(mesh->tangents);
		mMeshes[idx].indices.alloc_and_copy_from_host(mesh->indices);
		mMeshes[idx].material =
			mesh->getMaterial() ? &mMaterials[mesh->getMaterial()->getMaterialId()] : nullptr;
		if (mesh->inside) 
			mMeshes[idx].mediumInterface.inside = mMediumStorage.getPointer(mesh->inside);
		if (mesh->outside)
			mMeshes[idx].mediumInterface.outside = mMediumStorage.getPointer(mesh->outside);
	}
}

void RTScene::uploadSceneInstanceData() {
	/* Upload instance data to device... */
	auto &instances = mScene.lock()->getMeshInstances();
	mInstances.resize(instances.size());
	for (size_t idx = 0; idx < instances.size(); idx++) {
		const auto &instance   = instances[idx];
		auto &instanceData	   = mInstances[idx];
		Affine3f transform	   = instance->getNode()->getGlobalTransform();
		instanceData.transform = Transformation(transform);
		instanceData.mesh = &mMeshes[instance->getMesh()->getMeshId()];
	}
}

void RTScene::uploadSceneMaterialData() {
	/* Upload texture and material data to device... */
	auto &materials = mScene.lock()->getMaterials();
	mMaterials.resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		const auto &material = materials[idx];
		mMaterials[idx].initializeFromHost(material);
	}
}

void RTScene::uploadSceneLightData() {
	auto createTrianglePrimitives = [](Mesh::SharedPtr mesh, rt::InstanceData* instance) 
		-> std::vector<Triangle> {
		uint nTriangles = mesh->indices.size();
		std::vector<Triangle> triangles;
		for (uint i = 0; i < nTriangles; i++) 
			triangles.push_back(Triangle(i, instance));
		return triangles;
	};

	/* Process mesh lights (diffuse area lights).
	   Mesh lights do not actually exists in the scene graph, since rasterization does 
	   not inherently support them. We simply bypass them with storage in mesh data. */
	uint nMeshes = mScene.lock()->getMeshes().size();
	for (const auto &instance : mScene.lock()->getMeshInstances()) {
		const auto &mesh			   = instance->getMesh();
		const auto &material		   = mesh->getMaterial();
		if ((material && material->hasEmission()) || mesh->Le.any()) {
			rt::MaterialData &materialData = mMaterials[material->getMaterialId()];
			rt::MeshData &meshData		   = mMeshes[mesh->getMeshId()];
			rt::TextureData &textureData = materialData.getTexture(Material::TextureType::Emissive);
			rt::InstanceData &instanceData = mInstances[instance->getInstanceId()];
			RGB Le = material->hasEmission()
							 ? RGB(textureData.getConstant()) : mesh->Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld", 
					 " constant emission(?): %f", mesh->indices.size(), luminance(Le));
			std::vector<Triangle> primitives =
				createTrianglePrimitives(mesh, &mInstances[instance->getInstanceId()]);
			size_t n_primitives				 = primitives.size();
			instanceData.primitives.alloc_and_copy_from_host(primitives);
			std::vector<rt::DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] =
					rt::DiffuseAreaLight(Shape(&instanceData.primitives[triId]), textureData, Le);
			}
			instanceData.lights.alloc_and_copy_from_host(lights);
			for (size_t triId = 0; triId < n_primitives; triId++) 
				mLightStorage.addPointer(&instanceData.lights[triId]);
		}
	}

	/* Process other lights (environment lights and those analytical ones). */
	for (auto light : mScene.lock()->getLights()) {
		auto transform = light->getNode()->getGlobalTransform();
		float sceneRadius = mScene.lock()->getBoundingBox().diagonal().norm();
		if (auto infiniteLight = std::dynamic_pointer_cast<InfiniteLight>(light)) {
			rt::TextureData textureData;
			textureData.initializeFromHost(infiniteLight->getTexture());
			mLightStorage.emplaceEntity<rt::InfiniteLight>(
				light, transform.rotation(), textureData, light->getScale(), sceneRadius);
		} else if (auto pointLight = std::dynamic_pointer_cast<PointLight>(light)) {
			mLightStorage.emplaceEntity<rt::PointLight>(
				light, transform.translation(), pointLight->getColor(), pointLight->getScale());
		} else if (auto directionalLight = std::dynamic_pointer_cast<DirectionalLight>(light)) {
			mLightStorage.emplaceEntity<rt::DirectionalLight>(
				light, transform.rotation(), directionalLight->getColor(),
				directionalLight->getScale(), sceneRadius);
		} else {
			Log(Error, "Unsupported light type not uploaded to device memory.");
		}
	}
	
	mLightStorage.addPointers(mScene.lock()->getLights());

	/* Upload main constant light buffer and light sampler. */
	Log(Info, "A total of %zd light(s) processed!", mLightStorage.getPointers().size());
	if (!mLightStorage.getPointers().size())
		Log(Error, "There's no light source in the scene! "
			"Image will be dark, and may even cause crash...");
	auto lightSampler = UniformLightSampler(mLightStorage.getPointers());
	mLightSamplerBuffer.alloc_and_copy_from_host(&lightSampler, 1);
	// [Workaround] Since the area light hit depends on light buffer pointed from instance...
	CUDA_SYNC_CHECK();
}

void RTScene::uploadSceneMediumData() {
	// For a medium, its index in mediumBuffer is the same as medium->getMediumId();
	cudaDeviceSynchronize();
	for (auto medium : mScene.lock()->getMedia()) {
		if (auto m = std::dynamic_pointer_cast<HomogeneousVolume>(medium)) {
			mMediumStorage.emplaceEntity<HomogeneousMedium>(medium, m->sigma_t, m->albedo, m->Le, m->g, KRR_DEFAULT_COLORSPACE);
		} else if (auto m = std::dynamic_pointer_cast<VDBVolume>(medium)) {
			if (std::dynamic_pointer_cast<NanoVDBGrid<float>>(m->densityGrid)) {
				mMediumStorage.emplaceEntity<NanoVDBMedium<float>>(medium,
					m->getNode()->getGlobalTransform(), m->sigma_t, m->albedo, m->g,
					std::move(*std::dynamic_pointer_cast<NanoVDBGrid<float>>(m->densityGrid)),
					m->temperatureGrid ? std::move(*m->temperatureGrid) : NanoVDBGrid<float>{},
					m->albedoGrid ? std::move(*m->albedoGrid) : NanoVDBGrid<Array3f>{},
					m->scale, m->LeScale, m->temperatureScale, m->temperatureOffset, KRR_DEFAULT_COLORSPACE);
				mMediumStorage.getData<NanoVDBMedium<float>>().back().initializeFromHost();
			} else if (std::dynamic_pointer_cast<NanoVDBGrid<Array3f>>(m->densityGrid)) {
				mMediumStorage.emplaceEntity<NanoVDBMedium<Array3f>>(medium,
					m->getNode()->getGlobalTransform(), m->sigma_t, m->albedo, m->g,
					std::move(*std::dynamic_pointer_cast<NanoVDBGrid<Array3f>>(m->densityGrid)),
					m->temperatureGrid ? std::move(*m->temperatureGrid) : NanoVDBGrid<float>{},
					m->albedoGrid ? std::move(*m->albedoGrid) : NanoVDBGrid<Array3f>{},
					m->scale, m->LeScale, m->temperatureScale, m->temperatureOffset, KRR_DEFAULT_COLORSPACE);
				mMediumStorage.getData<NanoVDBMedium<Array3f>>().back().initializeFromHost();
			} else {
				Log(Error, "Unsupported heterogeneous VDB medium data type");
				continue;
			}
		} else
			Log(Error, "Unknown medium type not uploaded to device memory.");
	}

	mMediumStorage.addPointers(mScene.lock()->getMedia());
	CUDA_SYNC_CHECK();
}

rt::SceneData RTScene::getSceneData() {
	rt::SceneData sceneData {};
	sceneData.meshes		 = mMeshes;
	sceneData.instances		 = mInstances;
	sceneData.materials		 = mMaterials;
	sceneData.lights		 = mLightStorage.getPointers();
	sceneData.infiniteLights = mLightStorage.getData<rt::InfiniteLight>();
	sceneData.lightSampler	 = mLightSamplerBuffer.data();
	return sceneData;
}

void RTScene::update() { 
	mOptixScene->update();
	updateSceneData();
}

// This routine should only be called by OptixBackend...
void RTScene::updateSceneData() {
	PROFILE("Update scene data");
	// Currently we only support updating instance transformations...
	static size_t lastUpdatedFrame = 0;
	auto lastUpdates = mScene.lock()->getSceneGraph()->getLastUpdateRecord();
	if ((lastUpdates.updateFlags & SceneGraphNode::UpdateFlags::SubgraphTransform)
		!= SceneGraphNode::UpdateFlags::None && lastUpdatedFrame < lastUpdates.frameIndex) {
		cudaDeviceSynchronize();
		auto &instances = mScene.lock()->getMeshInstances();
		for (size_t idx = 0; idx < instances.size(); idx++) {
			const auto &instance   = instances[idx];
			auto &instanceData	   = mInstances[idx];
			Affine3f transform	   = instance->getNode()->getGlobalTransform();
			instanceData.transform = Transformation(transform);
		}
		lastUpdatedFrame = lastUpdates.frameIndex;
	}
	bool materialsChanged{false};
	for (const auto &material : mScene.lock()->getMaterials()) {
		if (material->isUpdated()) {
			cudaDeviceSynchronize();
			materialsChanged |= material->isUpdated();
			rt::MaterialData &materialData = mMaterials[material->getMaterialId()];
			materialData.mBsdfType		   = material->mBsdfType;
			materialData.mMaterialParams   = material->mMaterialParams;
			materialData.mShadingModel	   = material->mShadingModel;
			material->setUpdated(false);
		}
	}
	lastUpdatedFrame = lastUpdates.frameIndex;

}

NAMESPACE_END(krr)