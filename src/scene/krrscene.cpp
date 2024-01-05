#include "importer.h"

KRR_NAMESPACE_BEGIN
using namespace importer;

bool SceneImporter::loadModel(const fs::path filepath, Scene::SharedPtr pScene,
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

bool SceneImporter::loadLight(Scene::SharedPtr pScene, SceneGraphNode::SharedPtr node,
	const json& params) {
	auto type = params.value("type", "infinite");
	float scale = params.value<float>("scale", 1.f);
	RGB color	= params.value<Array3f>("color", Array3f{1, 1, 1});
	SceneLight::SharedPtr light;
	if (type == "point") {
		light	   = std::make_shared<PointLight>(color, scale);
	} else if (type == "directional") {
		light	   = std::make_shared<DirectionalLight>(color, scale);
	} else if (type == "infinite") {
		light		 = std::make_shared<InfiniteLight>(color, scale);
		auto texture = params.value<string>("texture", "");
		if (!texture.empty()) {
			auto tex = Texture::createFromFile(texture);
			std::dynamic_pointer_cast<InfiniteLight>(light)->setTexture(tex);
		}
	} else {
		Log(Error, "Unsupported light type: %s", type.c_str());
		return false;
	}
	if (light) 
		pScene->getSceneGraph()->attachLeaf(node, light);
	else
		Log(Error, "Failed to import light from configuration: %s", params.dump().c_str());
	return light != nullptr;
}

bool SceneImporter::loadMedium(Scene::SharedPtr pScene, SceneGraphNode::SharedPtr node,
								const json& params) {
	auto type = params.value("type", "homogeneous");
	if (type == "homogeneous") {
		auto sigma_a = params.value<Array3f>("sigma_a", Array3f{1, 1, 1});
		auto sigma_s = params.value<Array3f>("sigma_s", Array3f{0, 0, 0});
		auto g		 = params.value<float>("g", 0.f);
		auto Le		 = params.value<Array3f>("Le", Array3f{0, 0, 0});
		auto aabb	 = params.value<AABB3f>("bound", AABB3f{0, 0});

		auto mesh		= std::make_shared<Mesh>();
		auto instance	= std::make_shared<MeshInstance>(mesh);
		auto volume		= std::make_shared<HomogeneousVolume>(sigma_a, sigma_s, g, Le);
		mesh->indices	= {{4, 2, 0}, {2, 7, 3}, {6, 5, 7}, {1, 7, 5}, {0, 3, 1}, {4, 1, 5},
						   {4, 6, 2}, {2, 6, 7}, {6, 4, 5}, {1, 3, 7}, {0, 2, 3}, {4, 0, 1}};
		mesh->positions = {
			{aabb.max()[0], aabb.max()[1], aabb.min()[2]},
			{aabb.max()[0], aabb.min()[1], aabb.min()[2]},
			{aabb.max()[0], aabb.max()[1], aabb.max()[2]},
			{aabb.max()[0], aabb.min()[1], aabb.max()[2]},
			{aabb.min()[0], aabb.max()[1], aabb.min()[2]},
			{aabb.min()[0], aabb.min()[1], aabb.min()[2]},
			{aabb.min()[0], aabb.max()[1], aabb.max()[2]},
			{aabb.min()[0], aabb.min()[1], aabb.max()[2]},
		};

		mesh->setName("Medium box");
		mesh->setMaterial(nullptr);
		mesh->setMedium(volume, nullptr);
		mesh->computeBoundingBox();
		pScene->addMesh(mesh);
		pScene->getSceneGraph()->attachLeaf(node, instance);
		pScene->getSceneGraph()->attachLeaf(node, volume);
		return true;
	} else if (type == "heterogeneous") {
		if (params.contains("file")) {
			return OpenVDBImporter().import(params.at("file"), pScene, node, params);
		} else {
			Log(Error, "Heterogeneous medium must have a file path!");
			return false;
		}
	} else {
			Log(Error, "Unsupported medium type: %s", type.c_str());
			return false;
	}
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
	auto sceneGraph = scene->getSceneGraph(); 
	node			= node ? node : sceneGraph->getRoot();
	if (!node) {
		node = std::make_shared<SceneGraphNode>();
		sceneGraph->setRoot(node);
	}

	if (scene->getConfig().empty()) {
		// [TODO] only set the config once... any better solution?
		scene->setConfig(j);
	}
	
	if (j.contains("environment")) {
		if (!scene) Log(Fatal, "Import a model before doing scene configurations!");
		string env	 = j["environment"].get<string>();
		auto texture = Texture::createFromFile(env);
		auto root	 = sceneGraph->getRoot();
		auto light	 = std::make_shared<InfiniteLight>(texture);
		sceneGraph->attachLeaf(node, light);
	}

	if (j.contains("camera")) {
		auto cameraContainer = std::make_shared<SceneGraphNode>("Camera Container");
		sceneGraph->attach(sceneGraph->getRoot(), cameraContainer);
		auto camera = std::make_shared<Camera>();
		from_json(j.value("camera", json{}), *camera);
		sceneGraph->attachLeaf(cameraContainer, camera);
		scene->setCameraController(std::make_shared<OrbitCameraController>(
			j.value("cameraController", OrbitCameraController{})));
		scene->getCameraController()->setCamera(camera);
		scene->setCamera(camera);  /* default camera */
	}

	if (j.contains("model")) {
		importNode(j.at("model"), scene, node, params);
	}

	if (j.contains("options")) {
		auto options = j.at("options");
		scene->setAnimated(options.value("animated", false));
	}
	return true; 
}

bool SceneImporter::importNode(const json &j, Scene::SharedPtr scene, 
	SceneGraphNode::SharedPtr node, const json& params) {
	json::value_t ctype = j.type();
	if (ctype == json::value_t::array) {
		for (auto model : j) 
			importNode(model, scene, node, params);
	} else if (ctype == json::value_t::object) {
		// recursively unwrap the config
		// maybe another json object, an array, or a string indicating the filepath of the model
		auto child	   = std::make_shared<SceneGraphNode>();
		auto name	   = j.value<string>("name", "");
		auto type	   = j.value<string>("type", "model");
		auto translate = j.value<Vector3f>("translate", Vector3f::Zero());
		auto rotate	   = j.value<Quaternionf>("rotate", Quaternionf::Identity());
		auto scale	   = j.value<Vector3f>("scale", Vector3f::Ones());
			
		child->setName(name);
		child->setScaling(scale);
		child->setRotation(rotate);
		child->setTranslation(translate);
		scene->getSceneGraph()->attach(node, child);

		if (j.contains("model"))
			importNode(j["model"], scene, child, j.value("params", json{}));
		else {
			// [TODO] support other types of leaf nodes
			if (type == "medium") {
				loadMedium(scene, child, j.value("params", json{}));
			} else if (type == "light") {
				loadLight(scene, child, j.value("params", json{}));
			} else {
				Log(Error, "Unsupported node type: %s", type.c_str());
			}
		}
	} else if (ctype == json::value_t::string) {
		loadModel(j.get<string>(), scene, node, params);	
	} else {
		Log(Error, "Unsupported model type: %s", string(j).c_str());
		return false;
	}
	return true;
}

KRR_NAMESPACE_END