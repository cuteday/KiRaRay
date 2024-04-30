#include "importer.h"
#include "device/context.h"
#include "render/spectrum.h"

NAMESPACE_BEGIN(krr)
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
	} else if (type == "spotlight") {
		auto innerConeAngle = params.value<float>("inner_cone", 30.f);
		auto outerConeAngle = params.value<float>("outer_cone", 45.f);
		light = std::make_shared<SpotLight>(color, scale, innerConeAngle, outerConeAngle);
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
	if (light) {
		node->setLeaf(light);
		if (params.contains("position")) {
			auto position = params.value<Vector3f>("position", Vector3f::Zero());
			light->setPosition(position);
		}
		if (params.contains("direction")) {
			auto direction = params.value<Vector3f>("direction", -Vector3f::UnitZ());
			light->setDirection(direction);
		}
	}
	else
		Log(Error, "Failed to import light from configuration: %s", params.dump().c_str());
	return light != nullptr;
}

bool SceneImporter::loadMedium(Scene::SharedPtr pScene, SceneGraphNode::SharedPtr node,
								const json& params) {
	auto type = params.value("type", "homogeneous");
	if (type == "homogeneous") {
		auto sigma_t = params.value<Array3f>("sigma_t", Array3f{1, 1, 1});
		auto albedo	 = params.value<Array3f>("albedo", Array3f{0.5, 0.5, 0.5});
		auto g		 = params.value<float>("g", 0.f);
		auto Le		 = params.value<Array3f>("Le", Array3f{0, 0, 0});
		auto aabb	 = params.value<AABB3f>("bound", AABB3f{0, 0});

		if (params.contains("sigma_a") || params.contains("sigma_s")) {
			// backward compatibility
			auto sigma_a = params.value<Array3f>("sigma_a", Array3f{0.5, 0.5, 0.5});
			auto sigma_s = params.value<Array3f>("sigma_s", Array3f{0.5, 0.5, 0.5});
			sigma_t		 = sigma_a + sigma_s;
			albedo		 = sigma_s / sigma_t;
		}

		auto volume = std::make_shared<HomogeneousVolume>(sigma_t, albedo, g, Le, aabb);
		pScene->getSceneGraph()->attachLeaf(node, volume);

		if (params.contains("bound")) {
			auto mesh		= std::make_shared<Mesh>();
			auto instance	= std::make_shared<MeshInstance>(mesh);
			
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
		}

		if (params.contains("meshes")) {
			for (auto m : params.at("meshes")) {
				std::string meshName = m.get<std::string>();
				for (auto mesh : pScene->getMeshes()) {
					if (mesh->getName() == meshName) 
						mesh->setMedium(volume, mesh->outside);
				}
			}
		}

		if (params.contains("meshes_outside")) {
			for (auto m : params.at("meshes_outside")) {
				std::string meshName = m.get<std::string>();
				for (auto mesh : pScene->getMeshes()) {
					if (mesh->getName() == meshName) 
						mesh->setMedium(mesh->inside, volume);
				}
			}
		}
	
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

void SceneImporter::loadMedia(const json& j, Scene::SharedPtr scene) {
	if (j.type() != json::value_t::array) {
		Log(Error, "Media must be an json array of a list of individual media!");
		return;
	}
	auto root			= scene->getSceneGraph()->getRoot();
	auto mediaContainer = std::make_shared<SceneGraphNode>("Material Container");
	scene->getSceneGraph()->attach(root, mediaContainer);
	for (auto m : j) {
		loadMedium(scene, mediaContainer, m.at("params"));
	}
}

void SceneImporter::loadMaterials(const json& j, Scene::SharedPtr scene) {
	if (j.type() != json::value_t::array) {
		Log(Error, "Materials must be an json array of a material list!");
		return;
	}
	std::vector<Material::SharedPtr> loadedMaterials;
	auto& alloc			 = gpContext->alloc;
	auto extractSpectrum = [&](const json& j) -> Spectra {
		if(j.type() == json::value_t::string)
			return Spectra::getNamed(j.get<string>());
		else if (j.type() == json::value_t::object) {
			auto type = j.value<string>("type", "null");
			if (type == "named_spectrum") {
				return Spectra::getNamed(j.value<std::string>("name", ""));
			} else if (type == "constant_spectrum") {
				auto value = j.value<float>("value", 1.5f);
				return alloc->new_object<ConstantSpectrum>(value);
			} else if (type == "cauchy_spectrum") {
				auto a = j.value<float>("a", 1.5);
				auto b = j.value<float>("b", 0);
				return alloc->new_object<CauchyIoRSpectrum>(a, b);
			} else if (type == "sellmeier_spectrum") {
				auto b = j.value<Array3f>("b", {});
				auto c = j.value<Array3f>("c", {});
				return alloc->new_object<SellmeierIoRSpectrum>(b, c);
			} else {
				Log(Error, "Unsupported spectrum type: %s", type.c_str());
				return nullptr;
			}
		}else {
			Log(Error, "Unsupported parameter type: %s", j.dump().c_str());
			return nullptr;
		}
	};

	for (auto m : j) {
		auto name = m.value<string>("name", "Untitled");

		std::shared_ptr<Material> material;
		auto overrideMaterial = std::find_if(scene->getMaterials().begin(), scene->getMaterials().end(),
						[&](const Material::SharedPtr &v) { return v->getName() == name; });
		if (overrideMaterial == scene->getMaterials().end()) {
			material = std::make_shared<Material>(name);
			loadedMaterials.push_back(material);
		} else material = *overrideMaterial;
		// load material parameters and optionally textures
		Log(Info, "Overriding existing material %s", name.c_str());
		auto params = m.value<json>("params", json{});
		auto& matParams = material->mMaterialParams;
		matParams.diffuse.head<3>()	   = params.value<Array3f>("diffuse", Array3f::Ones());
		matParams.specular.head<3>()   = params.value<Array3f>("specular", Array3f::Zero());
		matParams.specular[3]		   = 1 - params.value<float>("roughness", 1.f);
		matParams.specularTransmission = params.value<float>("specular_transmission", 0.f);
		matParams.anisotropic		   = params.value<float>("anisotropic", 0.f);
		if (params.contains("eta")) 
			switch (params.at("eta").type()) {
				case json::value_t::number_float:
					matParams.IoR = params.value<float>("eta", 1.5f);
					break;
				default:
					matParams.spectralEta = extractSpectrum(params.at("eta"));
					matParams.IoR		  = matParams.spectralEta.maxValue();
			}
		if (params.contains("k")) {
			matParams.spectralK = extractSpectrum(params.at("k"));
		}
		material->mBsdfType		= m.value<MaterialType>("bsdf", MaterialType::Diffuse);
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		material->mColorSpace	= m.value<ColorSpaceType>("color_space", ColorSpaceType::sRGB);
	}
	if (!loadedMaterials.empty()) {
		// create a material container within the scene graph
		auto root = scene->getSceneGraph()->getRoot();
		auto materialContainer = std::make_shared<SceneGraphNode>("Material Container");
		scene->getSceneGraph()->attach(root, materialContainer);
		for (auto pMaterial : loadedMaterials)
			scene->getSceneGraph()->attachLeaf(materialContainer, pMaterial, pMaterial->getName());
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

	if (j.contains("media")) {
		loadMedia(j.at("media"), scene);
	}

	if (j.contains("materials")) {
		loadMaterials(j.at("materials"), scene);
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
		auto name	   = j.value<string>("name", "Untitled");	// Wah wah world!
		auto type	   = j.value<string>("type", "model");
		auto translate = j.value<Vector3f>("translate", Vector3f::Zero());
		auto rotate	   = j.value<Quaternionf>("rotate", Quaternionf::Identity());
		auto scale	   = j.value<Vector3f>("scale", Vector3f::Ones());
		auto params	   = j.value<json>("params", json{});
			
		child->setName(name);
		child->setScaling(scale);
		child->setRotation(rotate);
		child->setTranslation(translate);
		scene->getSceneGraph()->attach(node, child);

		if (j.contains("model"))
			importNode(j["model"], scene, child, params);
		else {
			// [TODO] support other types of leaf nodes
			if (type == "medium") {
				loadMedium(scene, child, params);
			} else if (type == "light") {
				loadLight(scene, child, params);
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

NAMESPACE_END(krr)