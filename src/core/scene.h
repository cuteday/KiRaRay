#pragma once
#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "camera.h"
#include "kiraray.h"

KRR_NAMESPACE_BEGIN

using namespace io;

class Texture {
	Texture() = default;
	Texture(uint32_t* data) :pixels(data) {}

	uint32_t* pixels = nullptr;
	vec2i resolution = { 0, 0 };
};

class Mesh {
public:
	std::vector<vec3f> vertices;
	std::vector<vec3f> normals;
	std::vector<vec2f> texcoords;
	std::vector<vec3i> indices;

	Texture* texture;
	int texture_id;
};

/* The scene class is in poccess of components like camera, cameracontroller, etc.
 * The update(), eventHandler(), renderUI(), of them is called within this class;
 */
class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene() = default;
	~Scene() = default;

	// intialization and creation
	void createFromFile(const string filepath);
	void processMesh(aiMesh* mesh, aiMatrix4x4 transform);
	void traverseNode(aiNode* node, aiMatrix4x4 transform);

	// user input handler
	void onMouseEvent(const MouseEvent& mouseEvent);
	void onKeyEvent(const KeyboardEvent& keyEvent);

	void update();
	void renderUI();

	Camera::SharedPtr getCamera() { return mpCamera; }
	CameraController::SharedPtr getCameraController() { return mpCameraController; }

	void setCamera(Camera::SharedPtr camera) { mpCamera = camera; }
	void setCameraController(CameraController::SharedPtr cameraController) { mpCameraController = cameraController; }

	//std::vector<Mesh*> meshes;
	std::vector<Mesh> meshes;
	std::vector<Texture*> textures;

private:
	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;

	aiScene* mScene;
};

KRR_NAMESPACE_END