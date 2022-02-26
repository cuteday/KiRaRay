#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"

#include "window.h"
#include "scene.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	// other components
	mpEnvLight = EnvLight::SharedPtr(new EnvLight());
	mpCamera = Camera::SharedPtr(new Camera());
	mpCameraController = OrbitCameraController::SharedPtr(new OrbitCameraController(mpCamera));
}

bool Scene::update()
{
	bool hasChanges = false;
	hasChanges |= mpCameraController->update();   
	hasChanges |= mpCamera->update();
	hasChanges |= mpEnvLight->update();
	return mHasChanges = hasChanges;
}

void Scene::renderUI() {
	
	if (ui::CollapsingHeader("Scene")) {
		if (mpCamera && ui::CollapsingHeader("Camera")) {
			mpCamera->renderUI();
		}
		if (mpEnvLight && ui::CollapsingHeader("Environment")) {
			mpEnvLight->renderUI();
		}
	}
}

void Scene::toDevice()
{
	std::vector<MeshData> meshData;
	for (Material& material : materials)
		material.toDevice();
	for (Mesh& mesh : meshes) {
		mesh.toDevice();
		meshData.push_back(mesh.mMeshData);
	}
	mMaterialBuffer.alloc_and_copy_from_host(materials);
	mMeshBuffer.alloc_and_copy_from_host(meshData);
	mData.materials = (Material*)mMaterialBuffer.data();
	mData.meshes = (MeshData*)mMeshBuffer.data();
}

bool Scene::onMouseEvent(const MouseEvent& mouseEvent)
{
	if(mpCameraController && mpCameraController->onMouseEvent(mouseEvent))
		return true;
	return false;
}

bool Scene::onKeyEvent(const KeyboardEvent& keyEvent)
{
	if(mpCameraController && mpCameraController->onKeyEvent(keyEvent))
		return true;
	return false;
}


KRR_NAMESPACE_END