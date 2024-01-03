#include "window.h"
#include "scene.h"

#include "device/context.h"
#include "render/profiler/profiler.h"
#include "scene.h"
#include "vulkan/scene.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
	mGraph			  = std::make_shared<SceneGraph>();
	mCamera			  = std::make_shared<Camera>();
	mCameraController = std::make_shared<OrbitCameraController>(mCamera);
}

bool Scene::update(size_t frameIndex, double currentTime) {
	bool hasChanges = false;
	if (mEnableAnimation) mGraph->animate(currentTime);
	mGraph->update(frameIndex);
	if (mCameraController) hasChanges |= mCameraController->update();
	if (mCamera) hasChanges |= mCamera->update();
	if (mSceneRT) mSceneRT->update();
	if (mSceneVK) mSceneVK->update();
	return mHasChanges = hasChanges;
}

void Scene::setCamera(Camera::SharedPtr camera) {
	mCamera = camera;
	camera->setScene(shared_from_this());
}

void Scene::setCameraController(OrbitCameraController::SharedPtr cameraController) {
	mCameraController = cameraController;
}

void Scene::renderUI() {
	if (ui::TreeNode("Statistics")) {
		ui::Text("Meshes: %d", getMeshes().size());
		ui::Text("Materials: %d", getMaterials().size());
		ui::Text("Instances: %d", getMeshInstances().size());
		ui::Text("Animations: %d", getAnimations().size());
		ui::Text("Media: %d", getMedia().size());
		ui::Text("Lights: %d", getLights().size());
		ui::TreePop();
	}
	if (mCamera && ui::TreeNode("Camera")) {
		ui::Text("Camera parameters");
		mCamera->renderUI();
		ui::Text("Orbit controller");
		mCameraController->renderUI();
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Scene Graph")) {
		mGraph->renderUI();
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Meshes")) {
		for (auto &mesh : getMeshes()) {
			if (ui::TreeNode(
					formatString("%d %s", mesh->getMeshId(), mesh->getName().c_str()).c_str())) {
				ui::PushID(mesh->getMeshId());
				mesh->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && ui::TreeNode("Materials")) {
		for (auto &material : getMaterials()) {
			if (ui::TreeNode(
					formatString("%d %s", material->getMaterialId(), material->getName().c_str())
						.c_str())) {
				ui::PushID(material->getMaterialId());
				material->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getMeshInstances().size() && ui::TreeNode("Instances")) {
		for (int i = 0; i < getMeshInstances().size(); i++) {
			if (ui::TreeNode(std::to_string(i).c_str())) {
				ui::PushID(getMeshInstances()[i]->getInstanceId());
				getMeshInstances()[i]->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getAnimations().size() && ui::TreeNode("Animations")) {
		ui::Checkbox("Enable animation", &mEnableAnimation);
		for (int i = 0; i < getAnimations().size(); i++) {
			if (ui::TreeNode(std::to_string(i).c_str())) {
				ui::PushID(i);
				getAnimations()[i]->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getLights().size() && ui::TreeNode("Lights")) {
		for (int i = 0; i < getLights().size(); i++) {
			auto light = getLights()[i];
			if (ui::TreeNode(light->getName().empty() ? std::to_string(i).c_str()
													  : light->getName().c_str())) {
				ui::PushID(i);
				light->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
	if (mGraph && getMedia().size() && ui::TreeNode("Media")) {
		for (int i = 0; i < getMedia().size(); i++) {
			auto medium = getMedia()[i];
			if (ui::TreeNode(medium->getName().empty() ? std::to_string(i).c_str()
													   : medium->getName().c_str())) {
				ui::PushID(i);
				medium->renderUI();
				ui::PopID();
				ui::TreePop();
			}
		}
		ui::TreePop();
	}
}

bool Scene::onMouseEvent(const MouseEvent& mouseEvent){
	if(mCameraController && mCameraController->onMouseEvent(mouseEvent))
		return true;
	return false;
}

bool Scene::onKeyEvent(const KeyboardEvent& keyEvent){
	if(mCameraController && mCameraController->onKeyEvent(keyEvent))
		return true;
	return false;
}

void Scene::initializeSceneRT() {
	if (!mGraph) Log(Fatal, "Scene graph must be initialized.");
	mGraph->update(0); // must be done before preparing device data.
	if (mSceneRT) {
		if (mSceneRT->getScene() == shared_from_this())
			Log(Debug, "The RT scene data has been initialized once before."
					 "I'm assuming you do not want to reinitialize it?");
		else Log(Error, "[Confused cat noise] A new scene?"
			"Currently only initialization with one scene is supported!");
		return;
	}
	mSceneRT = std::make_shared<RTScene>(shared_from_this()); 
	OptixSceneParameters optixSceneParameters{};
	if (mConfig.contains("options"))
		optixSceneParameters = mConfig["options"].get<OptixSceneParameters>();
	mSceneRT->uploadSceneData(optixSceneParameters);
}

void Scene::setConfig(const json& config, bool update) {
	if (update) mConfig.update(config);
	else mConfig = config;
	if (mConfig.contains("options")) {
		auto options = mConfig["options"];
		mEnableAnimation = options.value("animated", mEnableAnimation);
	}
}

KRR_NAMESPACE_END