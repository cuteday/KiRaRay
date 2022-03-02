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
	//for (Material& material : mData.materials)
	//	material.toDevice();
	mData.meshes.clear();
	for (Mesh& mesh : meshes) {
		mesh.toDevice();
		mData.meshes.push_back(mesh.mData);
	}
	//mData.nLights = mData.lights.size();
	
}

void Scene::processMeshLights()
{
	// This function should be called AFTER all scene data are moved to device
	uint nMeshes = meshes.size();
	for (uint meshId = 0; meshId < nMeshes; meshId++) {
		Mesh& mesh = meshes[meshId];
		Material& material = mData.materials[mesh.materialId];
		if (material.hasEmission()) {
			logDebug("Emissive diffuse area light detected,"
				" number of shapes: " + to_string(mesh.indices.size()) +
				" constant emission(?) "+ to_string(length(material.mMaterialParams.emissive)));
			std::vector<Triangle> triangles = mesh.createTriangles(&mData.meshes[meshId]);
			mesh.emissiveTriangles.assign(triangles.begin(), triangles.end());
			for (Triangle& tri : mesh.emissiveTriangles) {
				vec3f Le = material.mMaterialParams.emissive;
				Texture& texture = material.getTexture(Material::TextureType::Emissive);
				//DiffuseAreaLight light(Shape(&tri), texture, Le, true, 1.f);
				mesh.lights.push_back(DiffuseAreaLight(Shape(&tri), texture, Le, true, 1.f));
				mData.lights.push_back(&mesh.lights.back());
			}
			mesh.mData.lights = mesh.lights.data();
			mData.meshes[meshId] = mesh.mData;
		}
	}
	logDebug("Total " + to_string(mData.lights.size()) + " area lights processed!");
	mData.lightSampler = UniformLightSampler((inter::span<Light>)mData.lights);
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