#include "window.h"
#include "scene.h"

#include "device/context.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	mpCamera = Camera::SharedPtr(new Camera());
	mpCameraController = OrbitCameraController::SharedPtr(new OrbitCameraController(mpCamera));
}

bool Scene::update(){
	bool hasChanges = false;
	if (mpCameraController) hasChanges |= mpCameraController->update();
	if (mpCamera) hasChanges |= mpCamera->update();
	return mHasChanges = hasChanges;
}

void Scene::renderUI() {
	if (mpCamera && ui::CollapsingHeader("Camera")) {
		ui::Text("Camera parameters");
		mpCamera->renderUI();
		ui::Text("Orbit controller");
		mpCameraController->renderUI();
	}
	if (mpSceneRT && ui::CollapsingHeader("Ray-tracing Data")) {
		mpSceneRT->renderUI();
	}
}

void Scene::addEnvironmentMap(const Texture &infiniteLight) {
	environments.push_back(infiniteLight);
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
	mpSceneRT = std::make_shared<RTScene>(this); 
	mpSceneRT->toDevice();
}

void RTScene::toDevice() {
	cudaDeviceSynchronize();
	mDeviceData.materials = gpContext->alloc->new_object<inter::vector<rt::MaterialData>>();
	mDeviceData.meshes	= gpContext->alloc->new_object<inter::vector<rt::MeshData>>();
	mDeviceData.lights	= gpContext->alloc->new_object<inter::vector<Light>>();
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
	/* Upload mesh data to device... */
	auto meshes = mpScene->meshes;
	mDeviceData.meshes->resize(meshes.size());
	for (size_t idx = 0; idx < meshes.size(); idx++) {
		auto& mesh = mpScene->meshes[idx];
		auto &meshData = (*mDeviceData.meshes)[idx];
		meshData.vertices.alloc_and_copy_from_host(mesh.vertices);
		meshData.indices.alloc_and_copy_from_host(mesh.indices);
		meshData.materialId = mesh.materialId;
	}

	/* Upload texture and material data to device... */
	auto materials = mpScene->materials;
	mDeviceData.materials->resize(materials.size());
	for (size_t idx = 0; idx < materials.size(); idx++) {
		auto &material	   = mpScene->materials[idx];
		auto &materialData = (*mDeviceData.materials)[idx];
		materialData.initializeFromHost(material);
	}
	processLights();
}

void RTScene::processLights() {
	mDeviceData.lights->clear();
	cudaDeviceSynchronize();

	auto infiniteLights = mpScene->environments;
	mDeviceData.infiniteLights->reserve(infiniteLights.size());
	for (auto& infiniteLight : infiniteLights) {
		rt::TextureData textureData;
		textureData.initializeFromHost(infiniteLight);
		mDeviceData.infiniteLights->push_back(InfiniteLight(textureData));
	}

	uint nMeshes = mpScene->meshes.size();
	for (uint meshId = 0; meshId < nMeshes; meshId++) {
		Mesh &mesh		   = mpScene->meshes[meshId];
		Material &material = mpScene->materials[mesh.materialId];
		rt::MaterialData &materialData = (*mDeviceData.materials)[mesh.materialId];
		rt::MeshData &meshData = (*mDeviceData.meshes)[meshId];
		if (material.hasEmission() || mesh.Le.any()) {
			rt::TextureData &textureData =
				materialData.getTexture(Material::TextureType::Emissive);
			Color3f Le = material.hasEmission()
							 ? Color3f(textureData.getConstant()) : mesh.Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld", 
					 " constant emission(?): %f", mesh.indices.size(), luminance(Le));
			std::vector<Triangle> primitives = mesh.createTriangles(&meshData);
			size_t n_primitives				 = primitives.size();
			meshData.primitives.alloc_and_copy_from_host(primitives);
			meshData.lights.resize(primitives.size());
			std::vector<DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] = DiffuseAreaLight(Shape(&meshData.primitives[triId]),
									 textureData, Le, true, 1.f);
			}
			meshData.lights.alloc_and_copy_from_host(lights);
			for (size_t triId = 0; triId < n_primitives; triId++) 
				mDeviceData.lights->push_back(Light(&meshData.lights[triId]));
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