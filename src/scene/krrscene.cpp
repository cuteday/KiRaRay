#include "importer.h"

KRR_NAMESPACE_BEGIN
using namespace importer;

bool SceneImporter::loadScene(const fs::path filepath, Scene::SharedPtr pScene,
					  SceneGraphNode::SharedPtr node, const json &params) {
	bool success{};
	string format = filepath.extension().string();
	if (format == ".obj" || format == ".gltf" || format == ".glb" || format == ".fbx") {
		success = AssimpImporter().import(filepath, pScene, node, params);
	} else if (format == ".pbrt") {
		success = PbrtImporter().import(filepath, pScene, node, params);
	} else if (format == ".vdb" || format == ".nvdb") {
		success = OpenVDBImporter().import(filepath, pScene, node, params);
	} else if (format == ".json") {
		success = SceneImporter().import(filepath, pScene, node, params);
	} else {
		Log(Fatal, "Unsupported file format: %s...", format.c_str());
		return false;
	}
	if (!success) Log(Error, "Failed to load scene file from %ls...", filepath.c_str());
	return success;
}

bool SceneImporter::import(const fs::path filepath, Scene::SharedPtr scene,
						   SceneGraphNode::SharedPtr node, const json &params) {
	string format = filepath.extension().string();
	if (format == ".json") {
		json config = File::loadJSON(filepath);
		return import(config, scene, node);
	}
	else {
		Log(Fatal, "Unsupported format: %s", format.c_str());
		return false;
	}
}

bool SceneImporter::import(const json &j, Scene::SharedPtr scene, SceneGraphNode::SharedPtr node,
						   const json &params) { 
	node = node ? node : scene->getSceneGraph()->getRoot();
	if (!node) {
		node = std::make_shared<SceneGraphNode>();
		scene->getSceneGraph()->setRoot(node);
	}

	if (scene->getConfig().empty()) {
		// [TODO] only set the config once... any better solution?
		scene->setConfig(j);
	}
	
	if (j.contains("environment")) {
		if (!scene) Log(Fatal, "Import a model before doing scene configurations!");
		string env	 = j["environment"].get<string>();
		auto texture = Texture::createFromFile(env);
		auto root	 = scene->getSceneGraph()->getRoot();
		auto light	 = std::make_shared<InfiniteLight>(texture);
		scene->getSceneGraph()->attachLeaf(node, light);
	}

	if (j.contains("camera")) {
		scene->setCamera(std::make_shared<Camera>(j.at("camera")));
		scene->setCameraController(
			std::make_shared<OrbitCameraController>(j.at("cameraController")));
		scene->getCameraController()->setCamera(scene->mCamera);
	}

	if (j.contains("model")) {
		importModel(j.at("model"), scene, node, params);
	}
	return true; 
}

bool SceneImporter::importModel(const json &j, Scene::SharedPtr scene, 
	SceneGraphNode::SharedPtr node, const json& params) {
	json::value_t type = j.type();
	if (type == json::value_t::array) {
		for (auto model : j) 
			importModel(model, scene, node, params);
	} else if (type == json::value_t::object) {
		// recursively unwrap the config
		// maybe another json object, an array, or a string indicating the filepath of the model
		if (j.contains("model")) {
			auto child	   = std::make_shared<SceneGraphNode>();
			auto name	   = j.value<string>("name", "");
			auto translate = j.value<Vector3f>("translate", Vector3f::Zero());
			auto rotate	   = j.value<Quaternionf>("rotate", Quaternionf::Identity());
			auto scale	   = j.value<Vector3f>("scale", Vector3f::Ones());
			
			child->setName(name);
			child->setScaling(scale);
			child->setRotation(rotate);
			child->setTranslation(translate);
			scene->getSceneGraph()->attach(node, child);

			importModel(j["model"], scene, child, j.value("params", json{}));
		}
	} else if (type == json::value_t::string) {
		loadScene(j.get<string>(), scene, node, params);	
	} else {
		Log(Error, "Unsupported model type: %s", string(j).c_str());
		return false;
	}
	return true;
}

KRR_NAMESPACE_END