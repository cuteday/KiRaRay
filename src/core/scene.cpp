#include "window.h"
#include "scene.h"

#include "device/context.h"
#include "render/profiler/profiler.h"
#include "scene.h"
#include "vulkan/scene.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	mGraph			  = std::make_shared<SceneGraph>();
	mCamera			  = std::make_shared<Camera>();
	mCameraController = std::make_shared<OrbitCameraController>(mCamera);
}

bool Scene::update(size_t frameIndex, double currentTime) {
	bool hasChanges = false;
	if (mCameraController) hasChanges |= mCameraController->update();
	if (mCamera) hasChanges |= mCamera->update();
	if (mEnableAnimation) mGraph->animate(currentTime);
	mGraph->update(frameIndex);
	return mHasChanges = hasChanges;
}

void Scene::renderUI() {
	if (ui::TreeNode("Statistics")) {
		ui::Text("Meshes: %d", getMeshes().size());
		ui::Text("Materials: %d", getMaterials().size());
		ui::Text("Instances: %d", getMeshInstances().size());
		ui::Text("Animations: %d", getAnimations().size());
		ui::Text("Environment lights: %d", environments.size());
		ui::TreePop();
	}
	if (mCamera && ui::TreeNode("Camera")) {
		ui::Text("Camera parameters");
		mCamera->renderUI();
		ui::Text("Orbit controller");
		mCameraController->renderUI();
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Scene Graph")) {
		mGraph->renderUI();
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Meshes")) {
		for (auto &mesh : getMeshes()) {
			if (ui::TreeNode(formatString("%d %s", mesh->getMeshId(),
										  mesh->getName().c_str()).c_str())) {
				ui::PushID(mesh->getMeshId());
				mesh->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Materials")) {
		for (auto &material : getMaterials()) {
			if (ui::TreeNode(formatString("%d %s", material->getMaterialId(),
										  material->getName().c_str()).c_str())) {
				ui::PushID(material->getMaterialId());
				material->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getMeshInstances().size() && ui::TreeNode("Instances")) {
		for (int i = 0; i < getMeshInstances().size(); i++) {
			if (ui::TreeNode(std::to_string(i).c_str())) {
				ui::PushID(getMeshInstances()[i]->getInstanceId());
				getMeshInstances()[i]->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getAnimations().size() && ui::TreeNode("Animations")) {
		ui::Checkbox("Enable animation", &mEnableAnimation);
		for (int i = 0; i < getAnimations().size(); i++) {
			if (ui::TreeNode(std::to_string(i).c_str())) {
				getAnimations()[i]->renderUI();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
}

void Scene::addEnvironmentMap(Texture::SharedPtr infiniteLight) {
	environments.emplace_back(infiniteLight);
}

bool Scene::onMouseEvent(const MouseEvent& mouseEvent){
	if(mCameraController && mCameraController->onMouseEvent(mouseEvent))
		return true;
	return false;
}

bool Scene::onKeyEvent(const KeyboardEvent& keyEvent){
	if(mCameraController && mCameraController->onKeyEvent(keyEvent))
		return true;
	return false;
}

void Scene::initializeSceneRT() {
	if (!mGraph) Log(Fatal, "Scene graph must be initialized.");
	mGraph->update(0); // must be done before preparing device data.
	mSceneRT = std::make_shared<RTScene>(shared_from_this()); 
	mSceneRT->toDevice();
}

void RTScene::toDevice() {
	cudaDeviceSynchronize();
	uploadSceneData();
	CUDA_SYNC_CHECK();
}

void RTScene::uploadSceneData() {
	/* Upload texture and material data to device... */
	auto& materials = mScene.lock()->getMaterials();
	mMaterials.resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		const auto &material = materials[idx];
		mMaterials[idx].initializeFromHost(material);
	}
	mMaterialsBuffer.alloc_and_copy_from_host(mMaterials);

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
		mMeshes[idx].material = &mMaterialsBuffer[mesh->getMaterial()->getMaterialId()];
	}
	mMeshesBuffer.alloc_and_copy_from_host(mMeshes);

	/* Upload instance data to device... */
	auto &instances = mScene.lock()->getMeshInstances();
	mInstances.resize(instances.size());
	for (size_t idx = 0; idx < instances.size(); idx++) {
		const auto &instance	 = instances[idx];
		auto &instanceData		 = mInstances[idx];
		Affine3f transform		 = instance->getNode()->getGlobalTransform();
		instanceData.transform	 = transform;
		instanceData.transposedInverseTransform =
			transform.matrix().inverse().transpose().block<3, 3>(0, 0);
		instanceData.mesh		 = &mMeshesBuffer[instance->getMesh()->getMeshId()];
	}
	mInstancesBuffer.alloc_and_copy_from_host(mInstances);
	processLights();
	mInstancesBuffer.copy_from_host(mInstances.data(), mInstances.size());
}

void RTScene::processLights() {
	mLights.clear();

	auto createTrianglePrimitives = [](Mesh::SharedPtr mesh, rt::InstanceData* instance) 
		-> std::vector<Triangle> {
		uint nTriangles = mesh->indices.size();
		std::vector<Triangle> triangles;
		for (uint i = 0; i < nTriangles; i++) 
			triangles.push_back(Triangle(i, instance));
		return triangles;
	};

	auto infiniteLights = mScene.lock()->environments;
	mInfiniteLights.reserve(infiniteLights.size());
	for (auto& infiniteLight : infiniteLights) {
		rt::TextureData textureData;
		textureData.initializeFromHost(infiniteLight);
		mInfiniteLights.push_back(InfiniteLight(textureData));
	}
	mInfiniteLightsBuffer.alloc_and_copy_from_host(mInfiniteLights);

	uint nMeshes = mScene.lock()->getMeshes().size();
	for (const auto &instance : mScene.lock()->getMeshInstances()) {
		const auto &mesh			   = instance->getMesh();
		const auto &material		   = mesh->getMaterial();
		rt::MaterialData &materialData = mMaterials[material->getMaterialId()];
		rt::MeshData &meshData		   = mMeshes[mesh->getMeshId()];
		if (material->hasEmission() || mesh->Le.any()) {
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
			std::vector<DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] =
					DiffuseAreaLight(Shape(&instanceData.primitives[triId]), textureData, Le, true);
			}
			instanceData.lights.alloc_and_copy_from_host(lights);
			for (size_t triId = 0; triId < n_primitives; triId++) 
				mLights.push_back(Light(&instanceData.lights[triId]));
		}
	}
	for (int idx = 0; idx < mInfiniteLights.size(); idx++)
		mLights.push_back(Light(&mInfiniteLightsBuffer[idx]));
	mLightsBuffer.alloc_and_copy_from_host(mLights);
	Log(Info, "A total of %zd light(s) processed!", mLights.size());
	if (!mLights.size())
		Log(Error, "There's no light source in the scene! "
			"Image will be dark, and may even cause crash...");
	mLightSampler = UniformLightSampler(mLightsBuffer);
	mLightSamplerBuffer.alloc_and_copy_from_host(&mLightSampler, 1);
	CUDA_SYNC_CHECK();
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