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
class WavefrontPathTracer;
class OptiXBackend;
class OptiXWavefrontBackend;
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
	bool getChanges() { return mHasChanges; };
	void renderUI();
	void toDevice();

	void processLights();

	Camera& getCamera() { return *mpCamera; }
	CameraController& getCameraController() { return *mpCameraController; }

	void setCamera(Camera::SharedPtr camera) { mpCamera = camera; }
	void setCameraController(CameraController::SharedPtr cameraController) { mpCameraController = cameraController; }
	void addInfiniteLight(const InfiniteLight& infiniteLight);

	SceneData getSceneData() { return mData; }

private:
	friend class AssimpImporter;
	friend class PathTracer;
	friend class OptiXBackend;
	friend class OptiXWavefrontBackend;
	friend class WavefrontPathTracer;

	std::vector<Mesh> meshes;
	SceneData mData;
	Camera::SharedPtr mpCamera;
	CameraController::SharedPtr mpCameraController;
	bool mHasChanges = false;
};

KRR_NAMESPACE_END