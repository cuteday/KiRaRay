#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"

#include "window.h"
#include "scene.h"
#include "math/math.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	mpCamera = Camera::SharedPtr(new Camera());
	mpCameraController = OrbitCameraController::SharedPtr(new OrbitCameraController(mpCamera));
	assert(mData.materials == nullptr);
	mData.materials = gpContext->alloc->new_object<inter::vector<Material>>();
	mData.meshes = gpContext->alloc->new_object<inter::vector<MeshData>>();
	mData.lights = gpContext->alloc->new_object<inter::vector<Light>>();
	mData.infiniteLights = gpContext->alloc->new_object<inter::vector<InfiniteLight>>();
	mData.lightSampler = gpContext->alloc->new_object<UniformLightSampler>((inter::span<Light>)*mData.lights);
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
	if (ui::CollapsingHeader("Environment lights")) {
		CUDA_SYNC_CHECK();
		for (int i = 0; i < mData.infiniteLights->size(); i++) {
			if (ui::CollapsingHeader(to_string(i).c_str())) {
				(*mData.infiniteLights)[i].renderUI();
			}
		}
	}
	if (ui::CollapsingHeader("Materials")) {
		CUDA_SYNC_CHECK();
		for (int i = 0; i < mData.materials->size(); i++) {
			if (ui::CollapsingHeader((*mData.materials)[i].getName().c_str())) {
				(*mData.materials)[i].renderUI();
			}
		}
	}
}

void Scene::toDevice(){
	mData.meshes->clear();
	for (Mesh& mesh : meshes) {
		mesh.toDevice();
		mData.meshes->push_back(mesh.mData);
	}
	processLights();
}

void Scene::processLights(){
	// This function should be called AFTER all scene data are moved to device
	mData.lights->clear();
	cudaDeviceSynchronize();
	uint nMeshes = meshes.size();
	for (uint meshId = 0; meshId < nMeshes; meshId++) {
		Mesh& mesh = meshes[meshId];
		Material& material = (*mData.materials)[mesh.materialId];
		if (material.hasEmission() || mesh.Le.any()) {
			Texture &texture = material.getTexture(Material::TextureType::Emissive);
			Color3f Le		 = material.hasEmission() ? Color3f(texture.getConstant()) : mesh.Le;
			logDebug("Emissive diffuse area light detected,"
					 " number of shapes: " +
					 to_string(mesh.indices.size()) + " constant emission(?) " +
					 to_string(luminance(Le)));
			std::vector<Triangle> triangles = mesh.createTriangles(&(*mData.meshes)[meshId]);
			mesh.emissiveTriangles.assign(triangles.begin(), triangles.end());
			mesh.lights.clear();
			for (Triangle &tri : mesh.emissiveTriangles)
				mesh.lights.push_back(DiffuseAreaLight(Shape(&tri), texture, Le, true, 1.f));
			mesh.mData.lights		= mesh.lights.data();
			(*mData.meshes)[meshId] = mesh.mData;
		}
	}
	for (Mesh& mesh : meshes) {
		for (DiffuseAreaLight& light : mesh.lights)
			mData.lights->push_back(Light(&light));
	}
	for (InfiniteLight& light : *mData.infiniteLights)
		mData.lights->push_back(&light);
	logInfo("A total of " + to_string(mData.lights->size()) + " light(s) processed!");
	if (mData.lightSampler) 
		gpContext->alloc->deallocate_object((UniformLightSampler*)mData.lightSampler.ptr());
	mData.lightSampler = gpContext->alloc->new_object<UniformLightSampler>((inter::span<Light>) *mData.lights);
	CUDA_SYNC_CHECK();
}

void Scene::addInfiniteLight(const InfiniteLight& infiniteLight){
	mData.lights->clear();
	cudaDeviceSynchronize();
	mData.infiniteLights->push_back(infiniteLight);
	for (Mesh& mesh : meshes) {
		for (DiffuseAreaLight& light : mesh.lights)
			mData.lights->push_back(Light(&light));
	}
	for (InfiniteLight& light : *mData.infiniteLights)
		mData.lights->push_back(&light);
	if (mData.lightSampler.ptr()) 
		gpContext->alloc->deallocate_object((UniformLightSampler*)mData.lightSampler.ptr());
	mData.lightSampler = gpContext->alloc->new_object<UniformLightSampler>((inter::span<Light>) *mData.lights);
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


KRR_NAMESPACE_END