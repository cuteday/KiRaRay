#pragma once
#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "mesh.h"
#include "light.h"
#include "camera.h"
#include "envmap.h"
#include "texture.h"
#include "kiraray.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class AssimpImporter;
class PathTracer;
using namespace io;

/* The scene class is in poccess of components like camera, cameracontroller, etc.
 * The update(), eventHandler(), renderUI(), of them is called within this class;
 */
class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	struct SceneData {
		Material* materials;
		MeshData* meshes;
		TypedBuffer<Light> lights;
	};

	Scene();
	~Scene() = default;

	// user input handler
	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update();
	bool getChanges() { return mHasChanges; };
	void renderUI();
	void toDevice();

	Camera::SharedPtr getCamera() { return mpCamera; }
	CameraController::SharedPtr getCameraController() { return mpCameraController; }
	EnvLight::SharedPtr getEnvLight() { return mpEnvLight; }

	void setCamera(Camera::SharedPtr camera) { mpCamera = camera; }
	void setCameraController(CameraController::SharedPtr cameraController) { mpCameraController = cameraController; }
	void setEnvLight(EnvLight::SharedPtr envLight) { mpEnvLight = envLight; }
	SceneData getSceneData() { return mData; }

private:
	friend class AssimpImporter;
	friend class PathTracer;

	std::vector<Mesh> meshes;
	std::vector<Material> materials;

	SceneData mData;
	CUDABuffer mMeshBuffer;
	CUDABuffer mMaterialBuffer;
	TypedBuffer<Light> mLightBuffer;
	EnvLight::SharedPtr mpEnvLight;
	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;
	
	bool mHasChanges = false;
};

KRR_NAMESPACE_END