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
	if (mGraph && ui::TreeNode("Scene Animation")) {
		ui::Checkbox("Enable animation", &mEnableAnimation);
		for (int i = 0; i < getAnimations().size(); i++) {
			if (ui::TreeNode(std::to_string(i).c_str())) {
				getAnimations()[i]->renderUI();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mSceneRT && ui::TreeNode("Ray-tracing Data")) {
		mSceneRT->renderUI();
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
	if (!mDeviceData.materials)
		mDeviceData.materials = gpContext->alloc->new_object<inter::vector<rt::MaterialData>>();
	if (!mDeviceData.meshes)
		mDeviceData.meshes	= gpContext->alloc->new_object<inter::vector<rt::MeshData>>();
	if (!mDeviceData.instances)
		mDeviceData.instances = gpContext->alloc->new_object<inter::vector<rt::InstanceData>>();
	if (!mDeviceData.lights)
		mDeviceData.lights	= gpContext->alloc->new_object<inter::vector<Light>>();
	if (!mDeviceData.infiniteLights)
		mDeviceData.infiniteLights = gpContext->alloc->new_object<inter::vector<InfiniteLight>>();
	uploadSceneData();
	CUDA_SYNC_CHECK();
}

void RTScene::renderUI() {
	cudaDeviceSynchronize();

	if (ui::TreeNode("Environment lights")) {
		for (int i = 0; i < mDeviceData.infiniteLights->size(); i++) {
			if (ui::TreeNode(to_string(i).c_str())) {
				(*mDeviceData.infiniteLights)[i].renderUI();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (ui::TreeNode("Materials")) {
		for (int i = 0; i < mDeviceData.materials->size(); i++) {
			if (ui::TreeNode((*mDeviceData.materials)[i]
					.getHostMaterialPtr()->getName().c_str())) {
				(*mDeviceData.materials)[i].renderUI();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
}

void RTScene::uploadSceneData() {
	/* Upload texture and material data to device... */
	auto& materials = mScene.lock()->getMaterials();
	mDeviceData.materials->resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		const auto &material = materials[idx];
		auto &materialData	 = (*mDeviceData.materials)[idx];
		materialData.initializeFromHost(material);
	}

	/* Upload mesh data to device... */
	auto &meshes = mScene.lock()->getMeshes();
	mDeviceData.meshes->resize(meshes.size());
	for (size_t idx = 0; idx < meshes.size(); idx++) {
		const auto &mesh = meshes[idx];
		auto &meshData	 = (*mDeviceData.meshes)[idx];
		meshData.positions.alloc_and_copy_from_host(mesh->positions);
		meshData.normals.alloc_and_copy_from_host(mesh->normals);
		meshData.texcoords.alloc_and_copy_from_host(mesh->texcoords);
		meshData.tangents.alloc_and_copy_from_host(mesh->tangents);
		meshData.indices.alloc_and_copy_from_host(mesh->indices);	
		meshData.material = &(*mDeviceData.materials)[mesh->getMaterial()->getMaterialId()];
	}

	/* Upload instance data to device... */
	auto &instances = mScene.lock()->getMeshInstances();
	mDeviceData.instances->resize(instances.size());
	for (size_t idx = 0; idx < instances.size(); idx++) {
		const auto &instance	 = instances[idx];
		auto &instanceData		 = (*mDeviceData.instances)[idx];
		Affine3f transform		 = instance->getNode()->getGlobalTransform();
		instanceData.transform	 = transform;
		instanceData.transposedInverseTransform =
			transform.matrix().inverse().transpose().block<3, 3>(0, 0);
		instanceData.mesh		 = &(*mDeviceData.meshes)[instance->getMesh()->getMeshId()];
	}
	processLights();
}

void RTScene::processLights() {
	mDeviceData.lights->clear();
	cudaDeviceSynchronize();

	auto createTrianglePrimitives = [](Mesh::SharedPtr mesh, rt::InstanceData* instance) 
		-> std::vector<Triangle> {
		uint nTriangles = mesh->indices.size();
		std::vector<Triangle> triangles;
		for (uint i = 0; i < nTriangles; i++) 
			triangles.push_back(Triangle(i, instance));
		return triangles;
	};

	auto infiniteLights = mScene.lock()->environments;
	mDeviceData.infiniteLights->reserve(infiniteLights.size());
	for (auto& infiniteLight : infiniteLights) {
		rt::TextureData textureData;
		textureData.initializeFromHost(infiniteLight);
		mDeviceData.infiniteLights->push_back(InfiniteLight(textureData));
	}

	uint nMeshes = mScene.lock()->getMeshes().size();
	for (const auto &instance : mScene.lock()->getMeshInstances()) {
		const auto &mesh			   = instance->getMesh();
		const auto &material		   = mesh->getMaterial();
		rt::MaterialData &materialData = (*mDeviceData.materials)[material->getMaterialId()];
		rt::MeshData &meshData		   = (*mDeviceData.meshes)[mesh->getMeshId()];
		if (material->hasEmission() || mesh->Le.any()) {
			rt::TextureData &textureData = materialData.getTexture(Material::TextureType::Emissive);
			rt::InstanceData &instanceData = (*mDeviceData.instances)[instance->getInstanceId()];
			Color3f Le = material->hasEmission()
							 ? Color3f(textureData.getConstant()) : mesh->Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld", 
					 " constant emission(?): %f", mesh->indices.size(), luminance(Le));
			std::vector<Triangle> primitives = createTrianglePrimitives(mesh, &instanceData);
			size_t n_primitives				 = primitives.size();
			instanceData.primitives.alloc_and_copy_from_host(primitives);
			std::vector<DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] =
					DiffuseAreaLight(Shape(&instanceData.primitives[triId]), textureData, Le, true);
			}
			instanceData.lights.alloc_and_copy_from_host(lights);
			for (size_t triId = 0; triId < n_primitives; triId++) 
				mDeviceData.lights->push_back(Light(&instanceData.lights[triId]));
		}
	}
	for (InfiniteLight &light : *mDeviceData.infiniteLights)
		mDeviceData.lights->push_back(&light);
	Log(Info, "A total of %lld light(s) processed!", mDeviceData.lights->size());
	if (!mDeviceData.lights->size())
		Log(Error, "There's no light source in the scene! "
			"Image will be dark, and may even cause crash...");
	if (mDeviceData.lightSampler)
		gpContext->alloc->deallocate_object(
			(UniformLightSampler *) mDeviceData.lightSampler.ptr());
	mDeviceData.lightSampler =
		gpContext->alloc->new_object<UniformLightSampler>(
			(inter::span<Light>) *mDeviceData.lights);
	CUDA_SYNC_CHECK();
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
			auto &instanceData	   = (*mDeviceData.instances)[idx];
			Affine3f transform	   = instance->getNode()->getGlobalTransform();
			instanceData.transform = transform;
			instanceData.transposedInverseTransform =
				transform.matrix().inverse().transpose().block<3, 3>(0, 0);
		}
		lastUpdatedFrame = lastUpdates.frameIndex;
	}
}

KRR_NAMESPACE_END