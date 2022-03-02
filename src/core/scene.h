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
#include "interop.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "host/memory.h"
#include "render/lightsampler.h"

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
		inter::vector<Material> materials;
		inter::vector<MeshData> meshes;
		inter::vector<Light> lights;

		UniformLightSampler lightSampler;
	};

	Scene();
	~Scene() = default;

	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update();
	bool getChanges() { return mHasChanges; };
	void renderUI();
	void toDevice();

	void processMeshLights();

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
	SceneData mData;
	EnvLight::SharedPtr mpEnvLight;
	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;
	bool mHasChanges = false;
};

KRR_NAMESPACE_END