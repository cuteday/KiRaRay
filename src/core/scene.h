#pragma once
#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "camera.h"
#include "envmap.h"
//#include "texture.h"
#include "kiraray.h"

KRR_NAMESPACE_BEGIN

//class AssimpImporter;
using namespace io;

class Mesh {
public:
	struct MeshData{
		vec3f* vertices = nullptr;
		vec3i* indices = nullptr;
		vec3f* normals = nullptr;
		vec3f* texcoords = nullptr;
		vec3f* tangents = nullptr;
		vec3f* bitangents = nullptr;
	//	Material* material = nullptr;
	};

	std::vector<vec3f> vertices;
	std::vector<vec3f> normals;
	std::vector<vec2f> texcoords;
	std::vector<vec3i> indices;
	std::vector<vec3f> tangents;
	std::vector<vec3f> bitangents;

	//Material mMaterial;
	MeshData mDeviceMemory;
};
using MeshData = Mesh::MeshData;

/* The scene class is in poccess of components like camera, cameracontroller, etc.
 * The update(), eventHandler(), renderUI(), of them is called within this class;
 */
class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene();
	~Scene() = default;

	// user input handler
	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	void update();
	void renderUI();

	Camera::SharedPtr getCamera() { return mpCamera; }
	CameraController::SharedPtr getCameraController() { return mpCameraController; }
	EnvLight::SharedPtr getEnvLight() { return mpEnvLight; }

	void setCamera(Camera::SharedPtr camera) { mpCamera = camera; }
	void setCameraController(CameraController::SharedPtr cameraController) { mpCameraController = cameraController; }
	void setEnvLight(EnvLight::SharedPtr envLight) { mpEnvLight = envLight; }

	std::vector<Mesh> meshes;
//	std::vector<Material> materials;

private:
	//friend class AssimpImporter;

	EnvLight::SharedPtr mpEnvLight;

	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;

	aiScene* mScene;
};

KRR_NAMESPACE_END