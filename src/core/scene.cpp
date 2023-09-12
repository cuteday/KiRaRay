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
	if (mSceneRT) mSceneRT->update();
	if (mSceneVK) mSceneVK->update();
	return mHasChanges = hasChanges;
}

void Scene::loadConfig(const json& config) {
	mCamera			  = std::make_shared<Camera>(config.at("camera")); 
	mCameraController = std::make_shared<OrbitCameraController>(config.at("cameraController"));
	mCameraController->setCamera(mCamera);
}

void Scene::renderUI() {
	if (ui::TreeNode("Statistics")) {
		ui::Text("Meshes: %d", getMeshes().size());
		ui::Text("Materials: %d", getMaterials().size());
		ui::Text("Instances: %d", getMeshInstances().size());
		ui::Text("Animations: %d", getAnimations().size());
		ui::Text("Media: %d", getMedia().size());
		ui::Text("Lights: %d", getLights().size());
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
			if (ui::TreeNode(
					formatString("%d %s", mesh->getMeshId(), mesh->getName().c_str()).c_str())) {
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
			if (ui::TreeNode(
					formatString("%d %s", material->getMaterialId(), material->getName().c_str())
						.c_str())) {
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
				ui::PushID(i);
				getAnimations()[i]->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getLights().size() && ui::TreeNode("Lights")) {
		for (int i = 0; i < getLights().size(); i++) {
			auto light = getLights()[i];
			if (ui::TreeNode(light->getName().empty() ? std::to_string(i).c_str()
													  : light->getName().c_str())) {
				ui::PushID(i);
				light->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getMedia().size() && ui::TreeNode("Media")) {
		for (int i = 0; i < getMedia().size(); i++) {
			auto medium = getMedia()[i];
			if (ui::TreeNode(medium->getName().empty() ? std::to_string(i).c_str()
													   : medium->getName().c_str())) {
				ui::PushID(i);
				medium->renderUI();
				ui::PopID();
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
	if (mSceneRT) {
		if (mSceneRT->getScene() == shared_from_this())
			Log(Debug, "The RT scene data has been initialized once before."
					 "I'm assuming you do not want to reinitialize it?");
		else Log(Error, "[Confused cat noise] A new scene?"
			"Currently only initialization with one scene is supported!");
		return;
	}
	mSceneRT = std::make_shared<RTScene>(shared_from_this()); 
	mSceneRT->uploadSceneData();
}

RTScene::RTScene(Scene::SharedPtr scene) : mScene(scene) {}

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
		}
	}
	mHomogeneousMediumBuffer.alloc_and_copy_from_host(mHomogeneousMedium);

	size_t homogeneousId = 0;
	for (auto medium : mScene.lock()->getMedia()) {
		if (auto m = std::dynamic_pointer_cast<HomogeneousVolume>(medium)) 
			mMedium.push_back(Medium(&mHomogeneousMediumBuffer[homogeneousId++]));
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