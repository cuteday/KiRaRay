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

using namespace io;

typedef struct {
	MeshData* mesh;
} HitgroupSBTData;

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	HitgroupSBTData data;
};

/* The scene class is in poccess of components like camera, cameracontroller, etc.
 * The update(), eventHandler(), renderUI(), of them is called within this class;
 */
class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	struct SceneData {
		inter::vector<Material>* materials{ };
		inter::vector<MeshData>* meshes{ };
		inter::vector<Light>* lights{ };
		inter::vector<InfiniteLight>* infiniteLights{ };
		LightSampler lightSampler;
	};

	Scene();
	~Scene() = default;

	bool onMouseEvent(const MouseEvent& mouseEvent);
	bool onKeyEvent(const KeyboardEvent& keyEvent);

	bool update();
	bool getChanges() const { return mHasChanges; };
	void renderUI();
	void toDevice();

	void processLights();

	Camera& getCamera() { return *mpCamera; }
	CameraController& getCameraController() { return *mpCameraController; }

	void setCamera(const Camera &camera) { *mpCamera = camera; }
	void setCameraController(const OrbitCameraController &cameraController) {
		*mpCameraController = cameraController;
	}
	void addInfiniteLight(const InfiniteLight& infiniteLight);
	void loadConfig(const json &config) { 
		mpCamera = std::make_shared<Camera>(config.at("camera")); 
		mpCameraController = std::make_shared<OrbitCameraController>(config.at("cameraController"));
		mpCameraController->setCamera(mpCamera);
	}

	SceneData getSceneData() const { return mData; }
	AABB getAABB() const { return mAABB; }

	friend void to_json(json& j, const Scene& scene) { 
		j = json{ 
			{ "camera", *scene.mpCamera }, 
			{ "cameraController", *std::dynamic_pointer_cast<OrbitCameraController>(scene.mpCameraController) }
		};
	}

	std::vector<Mesh> meshes;
	SceneData mData;
	Camera::SharedPtr mpCamera;
	OrbitCameraController::SharedPtr mpCameraController;
	AABB mAABB;
	bool mHasChanges = false;
};

KRR_NAMESPACE_END