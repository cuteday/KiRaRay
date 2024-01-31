#pragma once
#include <map>

#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include <pbrtParser/Scene.h>

#include "common.h"
#include "scene.h"
#include "texture.h"

NAMESPACE_BEGIN(krr)

namespace importer {

class MaterialLoader {
public:
	using TextureType = Material::TextureType;

	MaterialLoader(bool useSrgb = true) { mUseSrgb = useSrgb; };
	~MaterialLoader(){};
	void loadTexture(const Material::SharedPtr &pMaterial, TextureType type,
					 const std::string &filename, bool flip = false);
	void setSrgb(bool useSrgb) { mUseSrgb = useSrgb; }

private:
	bool mUseSrgb{};
	using TextureKey = std::pair<std::string, bool>; // filename, srgb
	std::map<TextureKey, Texture::SharedPtr> mTextureCache;
};

class AssimpImporter {
public:
	enum class ImportMode {
		Default = 0,
		OBJ,
		GLTF2,
	};
	AssimpImporter() = default;
	bool import(const fs::path filepath, Scene::SharedPtr pScene,
				SceneGraphNode::SharedPtr node = nullptr, const json &params = json::object());

private:
	AssimpImporter(const AssimpImporter &) = delete;
	void operator=(const AssimpImporter &) = delete;

	void traverseNode(aiNode *assimnode, SceneGraphNode::SharedPtr graphNode);
	void loadMaterials(const string &modelFolder);
	void loadMeshes();
	void loadCameras();
	void loadAnimations();

	ImportMode mImportMode = ImportMode::Default;
	std::map<string, SceneGraphNode::SharedPtr> mNodeMap; // unique node name
	string mFilepath;
	aiScene *mAiScene = nullptr;
	Scene::SharedPtr mScene;
};

class PbrtImporter {
public:
	PbrtImporter() = default;
	bool import(const fs::path filepath, Scene::SharedPtr scene,
				SceneGraphNode::SharedPtr node = nullptr, const json &params = json::object());

private:
	PbrtImporter(const PbrtImporter &)	 = delete;
	void operator=(const PbrtImporter &) = delete;

	Mesh::SharedPtr loadMesh(pbrt::TriangleMesh::SP pbrtMesh);
	Material::SharedPtr loadMaterial(pbrt::Material::SP pbrtMaterial);
	Volume::SharedPtr loadMedium(pbrt::Medium::SP pbrtMedium);

	string resolve(string path);

	string mFilepath;
	string mBasepath;
	Scene::SharedPtr mScene;
};

class OpenVDBImporter {
public:
	OpenVDBImporter() = default;
	bool import(const fs::path filepath, Scene::SharedPtr scene,
				SceneGraphNode::SharedPtr node = nullptr, const json &params = json::object());

private:
	OpenVDBImporter(const OpenVDBImporter &) = delete;
	void operator=(const OpenVDBImporter &)	 = delete;

	string mFilepath;
	Scene::SharedPtr mScene;
};
} // namespace importer

class SceneImporter {
public:
	SceneImporter() = default;
	bool import(const fs::path filepath, Scene::SharedPtr scene,
				SceneGraphNode::SharedPtr node = nullptr, const json &params = json::object());
	bool import(const json &j, Scene::SharedPtr scene, SceneGraphNode::SharedPtr node = nullptr,
				const json &params = json::object());
	bool importNode(const json &j, Scene::SharedPtr scene,
					 SceneGraphNode::SharedPtr node = nullptr, const json &params = json::object());

	static bool loadModel(const fs::path filepath, Scene::SharedPtr pScene,
						  SceneGraphNode::SharedPtr node = nullptr,
						  const json &params			 = json::object());
	static bool loadMedium(Scene::SharedPtr pScene, SceneGraphNode::SharedPtr node = nullptr,
						   const json &params = json::object());
	static bool loadLight(Scene::SharedPtr pScene, SceneGraphNode::SharedPtr node = nullptr,
						   const json &params = json::object());


private:
	SceneImporter(const SceneImporter &)  = delete;
	void operator=(const SceneImporter &) = delete;

	string mFilepath;
	Scene::SharedPtr mScene;
};

NAMESPACE_END(krr)