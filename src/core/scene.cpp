#include "window.h"
#include "scene.h"

#include "device/context.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	mGraph			   = std::make_shared<SceneGraph>();
	mpCamera		   = std::make_shared<Camera>();
	mpCameraController = std::make_shared<OrbitCameraController>(mpCamera);
}

bool Scene::update(size_t frameIndex){
	bool hasChanges = false;
	if (mpCameraController) hasChanges |= mpCameraController->update();
	if (mpCamera) hasChanges |= mpCamera->update();
	mGraph->update(frameIndex);
	return mHasChanges = hasChanges;
}

void Scene::renderUI() {
	if (ui::CollapsingHeader("Statistics")) {
		ui::Text("Meshes: %d", getMeshes().size());
		ui::Text("Materials: %d", getMaterials().size());
		ui::Text("Instances: %d", getMeshInstances().size());
		ui::Text("Environment lights: %d", environments.size());
	}
	if (mpCamera && ui::CollapsingHeader("Camera")) {
		ui::Text("Camera parameters");
		mpCamera->renderUI();
		ui::Text("Orbit controller");
		mpCameraController->renderUI();
	}
	if (mGraph && ui::CollapsingHeader("Scene Graph")) {
		mGraph->renderUI();
	}
	if (mpSceneRT && ui::CollapsingHeader("Ray-tracing Data")) {
		mpSceneRT->renderUI();
	}
}

void Scene::addEnvironmentMap(Texture::SharedPtr infiniteLight) {
	environments.emplace_back(infiniteLight);
}

bool Scene::onMouseEvent(const MouseEvent& mouseEvent){
	if(mpCameraController && mpCameraController->onMouseEvent(mouseEvent))
		return true;
	return false;
}

bool Scene::onKeyEvent(const KeyboardEvent& keyEvent){
	if(mpCameraController && mpCameraController->onKeyEvent(keyEvent))
		return true;
	return false;
}

void Scene::initializeSceneRT() { 
	update(0);				// must be done before preparing device data.
	mpSceneRT = std::make_shared<RTScene>(this); 
	mpSceneRT->toDevice();
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
	updateSceneData();
	CUDA_SYNC_CHECK();
}

void RTScene::renderUI() {
	cudaDeviceSynchronize();

	if (ui::CollapsingHeader("Environment lights")) {
		for (int i = 0; i < mDeviceData.infiniteLights->size(); i++) {
			if (ui::CollapsingHeader(to_string(i).c_str())) {
				(*mDeviceData.infiniteLights)[i].renderUI();
			}
		}
	}
	if (ui::CollapsingHeader("Materials")) {
		for (int i = 0; i < mDeviceData.materials->size(); i++) {
			if (ui::CollapsingHeader((*mDeviceData.materials)[i]
					.getHostMaterialPtr()->getName().c_str())) {
				(*mDeviceData.materials)[i].renderUI();
			}
		}
	}
}

void RTScene::updateSceneData() {
	/* Upload texture and material data to device... */
	auto& materials = mpScene->getMaterials();
	mDeviceData.materials->resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		const auto &material = materials[idx];
		auto &materialData	 = (*mDeviceData.materials)[idx];
		materialData.initializeFromHost(material);
	}

	/* Upload mesh data to device... */
	auto &meshes = mpScene->getMeshes();
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
	auto &instances = mpScene->getMeshInstances();
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

	auto infiniteLights = mpScene->environments;
	mDeviceData.infiniteLights->reserve(infiniteLights.size());
	for (auto& infiniteLight : infiniteLights) {
		rt::TextureData textureData;
		textureData.initializeFromHost(infiniteLight);
		mDeviceData.infiniteLights->push_back(InfiniteLight(textureData));
	}

	uint nMeshes = mpScene->getMeshes().size();
	for (const auto &instance: mpScene->getMeshInstances()) {
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

KRR_NAMESPACE_END