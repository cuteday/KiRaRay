#pragma once
#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "camera.h"
#include "envmap.h"
#include "texture.h"
#include "kiraray.h"
#include "gpu/buffer.h"

KRR_NAMESPACE_BEGIN

class AssimpImporter;
using namespace io;

class Mesh {
public:
	struct MeshData{
		vec3f* vertices = nullptr;
		vec3i* indices = nullptr;
		vec3f* normals = nullptr;
		vec2f* texcoords = nullptr;
		vec3f* tangents = nullptr;
		vec3f* bitangents = nullptr;
		uint materialId = 0;
	};

	struct DeviceMemory {
		CUDABuffer vertices;
		CUDABuffer normals;
		CUDABuffer texcoords;
		CUDABuffer indices;
		CUDABuffer tangents;
		CUDABuffer bitangents;
		CUDABuffer material;
	};

	void toDevice() {
		mDeviceMemory.vertices.alloc_and_copy_from_host(vertices);
		mDeviceMemory.normals.alloc_and_copy_from_host(normals);
		mDeviceMemory.texcoords.alloc_and_copy_from_host(texcoords);
		mDeviceMemory.indices.alloc_and_copy_from_host(indices);
		mDeviceMemory.tangents.alloc_and_copy_from_host(tangents);
		mDeviceMemory.bitangents.alloc_and_copy_from_host(bitangents);

		mMeshData.vertices = (vec3f*)mDeviceMemory.vertices.data();
		mMeshData.normals = (vec3f*)mDeviceMemory.normals.data();
		mMeshData.indices = (vec3i*)mDeviceMemory.indices.data();
		mMeshData.texcoords = (vec2f*)mDeviceMemory.texcoords.data();
		mMeshData.tangents = (vec3f*)mDeviceMemory.tangents.data();
		mMeshData.bitangents = (vec3f*)mDeviceMemory.bitangents.data();
		mMeshData.materialId = mMaterialId;
	}

	std::vector<vec3f> vertices;
	std::vector<vec3f> normals;
	std::vector<vec2f> texcoords;
	std::vector<vec3i> indices;
	std::vector<vec3f> tangents;
	std::vector<vec3f> bitangents;

	uint mMaterialId = 0;
	MeshData mMeshData;
	DeviceMemory mDeviceMemory;
};
using MeshData = Mesh::MeshData;

/* The scene class is in poccess of components like camera, cameracontroller, etc.
 * The update(), eventHandler(), renderUI(), of them is called within this class;
 */
class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	struct SceneData {
		Material* materials;
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

	std::vector<Mesh> meshes;
	std::vector<Material> materials;

private:
	friend class AssimpImporter;

	SceneData mData;
	CUDABuffer mMaterialBuffer;
	EnvLight::SharedPtr mpEnvLight;
	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;
	
	bool mHasChanges = false;
};

KRR_NAMESPACE_END