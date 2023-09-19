#include "device/scene.h"
#include "device/optix.h"
#include "render/profiler/profiler.h"
#include "../scene.h"
#include "util/volume.h"

KRR_NAMESPACE_BEGIN

RTScene::RTScene(Scene::SharedPtr scene) : mScene(scene) {}

std::shared_ptr<Scene> RTScene::getScene() const { return mScene.lock(); }

void RTScene::uploadSceneData() {
	// The order of upload is unchangeable since some of them are dependent to others.
	uploadSceneMaterialData();
	uploadSceneMediumData();
	uploadSceneMeshData();
	uploadSceneInstanceData();
	uploadSceneLightData();

	CUDA_SYNC_CHECK();
	mOptixScene = std::make_shared<OptixScene>(mScene.lock());
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
			mesh->getMaterial() ? &mMaterialsBuffer[mesh->getMaterial()->getMaterialId()] : nullptr;
		if (mesh->inside) 
			mMeshes[idx].mediumInterface.inside = mMedium[mesh->inside->getMediumId()];
		if (mesh->outside)
			mMeshes[idx].mediumInterface.outside = mMedium[mesh->outside->getMediumId()];
	}
	mMeshesBuffer.alloc_and_copy_from_host(mMeshes);
}

void RTScene::uploadSceneInstanceData() {
	/* Upload instance data to device... */
	auto &instances = mScene.lock()->getMeshInstances();
	mInstances.resize(instances.size());
	for (size_t idx = 0; idx < instances.size(); idx++) {
		const auto &instance   = instances[idx];
		auto &instanceData	   = mInstances[idx];
		Affine3f transform	   = instance->getNode()->getGlobalTransform();
		instanceData.transform = transform;
		instanceData.transposedInverseTransform =
			transform.matrix().inverse().transpose().block<3, 3>(0, 0);
		instanceData.mesh = &mMeshesBuffer[instance->getMesh()->getMeshId()];
	}
	mInstancesBuffer.alloc_and_copy_from_host(mInstances);
}

void RTScene::uploadSceneMaterialData() {
	/* Upload texture and material data to device... */
	auto &materials = mScene.lock()->getMaterials();
	mMaterials.resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		const auto &material = materials[idx];
		mMaterials[idx].initializeFromHost(material);
	}
	mMaterialsBuffer.alloc_and_copy_from_host(mMaterials);
}

void RTScene::uploadSceneLightData() {
	mLights.clear();

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
		rt::MaterialData &materialData = mMaterials[material->getMaterialId()];
		rt::MeshData &meshData		   = mMeshes[mesh->getMeshId()];
		if ((material && material->hasEmission()) || mesh->Le.any()) {
			rt::TextureData &textureData = materialData.getTexture(Material::TextureType::Emissive);
			rt::InstanceData &instanceData = mInstances[instance->getInstanceId()];
			Color3f Le = material->hasEmission()
							 ? Color3f(textureData.getConstant()) : mesh->Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld", 
					 " constant emission(?): %f", mesh->indices.size(), luminance(Le));
			std::vector<Triangle> primitives =
				createTrianglePrimitives(mesh, &mInstancesBuffer[instance->getInstanceId()]);
			size_t n_primitives				 = primitives.size();
			instanceData.primitives.alloc_and_copy_from_host(primitives);
			std::vector<rt::DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] =
					rt::DiffuseAreaLight(Shape(&instanceData.primitives[triId]), textureData, Le);
			}
			instanceData.lights.alloc_and_copy_from_host(lights);
			for (size_t triId = 0; triId < n_primitives; triId++) 
				mLights.push_back(rt::Light(&instanceData.lights[triId]));
		}
	}

	/* Process other lights (environment lights and those analytical ones). */
	for (auto light : mScene.lock()->getLights()) {
		auto transform = light->getNode()->getGlobalTransform();
		if (auto infiniteLight = std::dynamic_pointer_cast<InfiniteLight>(light)) {
			rt::TextureData textureData;
			textureData.initializeFromHost(infiniteLight->getTexture());
			mInfiniteLights.push_back(rt::InfiniteLight(transform.rotation(), textureData, 1));
		} else if (auto pointLight = std::dynamic_pointer_cast<PointLight>(light)) {
			Log(Warning, "Point light is not yet implemented in ray tracing, skipping...");
		} else if (auto directionalLight = std::dynamic_pointer_cast<DirectionalLight>(light)) {
			Log(Warning, "Directional light is not yet implemented in ray tracing, skipping...");
		}
	}

	/* Upload infinite lights (a.k.a. environment lights). */
	mInfiniteLightsBuffer.alloc_and_copy_from_host(mInfiniteLights);
	for (int idx = 0; idx < mInfiniteLights.size(); idx++)
		mLights.push_back(rt::Light(&mInfiniteLightsBuffer[idx]));

	/* Upload main constant light buffer and light sampler. */
	mLightsBuffer.alloc_and_copy_from_host(mLights);
	Log(Info, "A total of %zd light(s) processed!", mLights.size());
	if (!mLights.size())
		Log(Error, "There's no light source in the scene! "
			"Image will be dark, and may even cause crash...");
	mLightSampler = UniformLightSampler(mLightsBuffer);
	mLightSamplerBuffer.alloc_and_copy_from_host(&mLightSampler, 1);
	// [Workaround] Since the area light hit depends on light buffer pointed from instance...
	mInstancesBuffer.alloc_and_copy_from_host(mInstances);
	CUDA_SYNC_CHECK();
}

void RTScene::uploadSceneMediumData() {
	// For a medium, its index in mediumBuffer is the same as medium->getMediumId();
	for (auto medium : mScene.lock()->getMedia()) {
		if (auto m = std::dynamic_pointer_cast<HomogeneousVolume>(medium)) {
			HomogeneousMedium gMedium(m->sigma_a, m->sigma_s, m->Le, m->g);
			mHomogeneousMedium.push_back(gMedium);
		} else if (auto m = std::dynamic_pointer_cast<VDBVolume>(medium)) {
			float maxDensity{0};
			auto densityGridHandle = loadNanoVDB(m->densityFile, &maxDensity);
			densityGridHandle.deviceUpload();
			mNanoVDBMedium.emplace_back(m->getNode()->getGlobalTransform(), m->sigma_a, m->sigma_s, m->g, 
				NanoVDBGrid(std::move(densityGridHandle), maxDensity));
		}
	}
	mHomogeneousMediumBuffer.alloc_and_copy_from_host(mHomogeneousMedium);
	mNanoVDBMediumBuffer.alloc_and_copy_from_host(mNanoVDBMedium);

	size_t homogeneousId = 0;
	size_t nanoVDBId	 = 0;
	for (auto medium : mScene.lock()->getMedia()) {
		if (auto m = std::dynamic_pointer_cast<HomogeneousVolume>(medium)) 
			mMedium.push_back(Medium(&mHomogeneousMediumBuffer[homogeneousId++]));
		else if (auto m = std::dynamic_pointer_cast<VDBVolume>(medium))
			mMedium.push_back(Medium(&mNanoVDBMediumBuffer[nanoVDBId++]));
		else Log(Error, "Unknown medium type not uploaded to device memory.");
	}
	mMediumBuffer.alloc_and_copy_from_host(mMedium);
}

rt::SceneData RTScene::getSceneData() const {
	rt::SceneData sceneData {};
	sceneData.meshes		 = mMeshesBuffer;
	sceneData.instances		 = mInstancesBuffer;
	sceneData.materials		 = mMaterialsBuffer;
	sceneData.lights		 = mLightsBuffer;
	sceneData.infiniteLights = mInfiniteLightsBuffer;
	sceneData.lightSampler	 = mLightSamplerBuffer.data();
	return sceneData;
}

void RTScene::update() { 
	static size_t lastUpdatedFrame = 0;
	auto lastUpdates = mScene.lock()->getSceneGraph()->getLastUpdateRecord();
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
		auto &instances = mScene.lock()->getMeshInstances();
		for (size_t idx = 0; idx < instances.size(); idx++) {
			const auto &instance   = instances[idx];
			auto &instanceData	   = mInstances[idx];
			Affine3f transform	   = instance->getNode()->getGlobalTransform();
			instanceData.transform = transform;
			instanceData.transposedInverseTransform =
				transform.matrix().inverse().transpose().block<3, 3>(0, 0);
		}
		mInstancesBuffer.copy_from_host(mInstances.data(), mInstances.size());
		lastUpdatedFrame = lastUpdates.frameIndex;
	}
	bool materialsChanged{false};
	for (const auto &material : mScene.lock()->getMaterials()) {
		if (material->isUpdated()) {
			materialsChanged |= material->isUpdated();
			rt::MaterialData &materialData = mMaterials[material->getMaterialId()];
			materialData.mBsdfType		   = material->mBsdfType;
			materialData.mMaterialParams   = material->mMaterialParams;
			materialData.mShadingModel	   = material->mShadingModel;
			material->setUpdated(false);
		}
	}
	if (materialsChanged) 
		mMaterialsBuffer.copy_from_host(mMaterials.data(), mMaterials.size());	
	lastUpdatedFrame = lastUpdates.frameIndex;
}

KRR_NAMESPACE_END