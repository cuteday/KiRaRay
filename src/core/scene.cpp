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

void Scene::addInfiniteLight(const InfiniteLight& infiniteLight){
	infiniteLights.push_back(infiniteLight);
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
	mDeviceData.materials = gpContext->alloc->new_object<inter::vector<Material>>();
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
			if (ui::CollapsingHeader(
					(*mDeviceData.materials)[i].getName().c_str())) {
				(*mDeviceData.materials)[i].renderUI();
			}
		}
	}
}

void RTScene::updateSceneData() {
	auto meshes = mpScene->meshes;
	
	/* Upload mesh data to device... */
	mDeviceData.meshes->resize(mpScene->meshes.size());
	for (size_t idx = 0; idx < mDeviceData.meshes->size(); idx++) {
		auto& mesh = mpScene->meshes[idx];
		auto &meshData = (*mDeviceData.meshes)[idx];
		meshData.vertices.alloc_and_copy_from_host(mesh.vertices);
		meshData.indices.alloc_and_copy_from_host(mesh.indices);
		meshData.materialId = mesh.materialId;
	}
	
	auto materials = mpScene->materials;
	mDeviceData.materials->assign(materials.begin(), materials.end());
	auto infiniteLights = mpScene->infiniteLights;
	mDeviceData.infiniteLights->assign(infiniteLights.begin(),
									   infiniteLights.end());
	processLights();
}

void RTScene::processLights() {
	mDeviceData.lights->clear();
	cudaDeviceSynchronize();
	uint nMeshes = mpScene->meshes.size();
	for (uint meshId = 0; meshId < nMeshes; meshId++) {
		Mesh &mesh		   = mpScene->meshes[meshId];
		Material &material = (*mDeviceData.materials)[mesh.materialId];
		rt::MeshData &meshData = (*mDeviceData.meshes)[meshId];
		if (material.hasEmission() || mesh.Le.any()) {
			Texture &texture =
				material.getTexture(Material::TextureType::Emissive);
			Color3f Le = material.hasEmission() ? Color3f(texture.getConstant())
												: mesh.Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld", 
					 " constant emission(?): %f", mesh.indices.size(), luminance(Le));
			std::vector<Triangle> primitives = mesh.createTriangles(&meshData);
			size_t n_primitives				 = primitives.size();
			meshData.primitives.alloc_and_copy_from_host(primitives);
			meshData.lights.resize(primitives.size());
			std::vector<DiffuseAreaLight> lights(n_primitives);
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] = DiffuseAreaLight(Shape(&meshData.primitives[triId]),
												 texture, Le, true, 1.f);
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