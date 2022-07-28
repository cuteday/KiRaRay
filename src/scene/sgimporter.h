#pragma once
#include <map>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"

#include "common.h"
#include "scene.h"
#include "scenegraph.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

class MaterialLoader {
public:
	using TextureType = Material::TextureType;

	MaterialLoader(bool useSrgb = true) { mUseSrgb = useSrgb; };
	~MaterialLoader(){};
	void loadTexture(const Material::SharedPtr &pMaterial, TextureType type,
					 const std::string &filename);
	void setSrgb(bool useSrgb) { mUseSrgb = useSrgb; }

private:
	bool mUseSrgb{};
	using TextureKey = std::pair<std::string, bool>; // filename, srgb
	std::map<TextureKey, Texture> mTextureCache;
};

class SceneGraphImporter {
public:
	enum class ImportMode {
		Default = 0,
		OBJ,
		GLTF2,
	};
	SceneGraphImporter() = default;
	bool import(const string &filepath, Scene::SharedPtr pScene);

private:
	SceneGroup::SharedPtr traverseNode(aiNode *node);
	
	void loadMaterials(const string &modelFolder);
	void loadMeshes();

	SceneGraphImporter(const SceneGraphImporter &) = delete;
	void operator=(const SceneGraphImporter &) = delete;

	ImportMode mImportMode = ImportMode::Default;
	std::vector<uint> mRemappedMeshIndices;
	string mFilepath;
	aiScene *mpAiScene = nullptr;
	Scene::SharedPtr mpScene;
};

KRR_NAMESPACE_END