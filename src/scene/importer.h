#pragma once
#include <map>

#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "scene.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

class MaterialLoader
{
public:
	using TextureType = Material::TextureType;

	MaterialLoader(bool useSrgb = true) { mUseSrgb = useSrgb; };
	~MaterialLoader() {};

	/** Request loading a material texture.
		\param[in] pMaterial Material to load texture into.
		\param[in] slot Slot to load texture into.
		\param[in] filename Texture filename.
	*/
	void loadTexture(const Material::SharedPtr& pMaterial, TextureType type, const std::string& filename);
	void setSrgb(bool useSrgb) { mUseSrgb = useSrgb; }

private:
	bool mUseSrgb;
	using TextureKey = std::pair<std::string, bool>; // filename, srgb
	std::map<TextureKey, Texture> mTextureCache;
};

class AssimpImporter {
public:
	enum class ImportMode {
		Default = 0,
		OBJ,
		GLTF2,
	};

	AssimpImporter() = default;
	bool import(const string& filepath, Scene::SharedPtr pScene);

private:

	void processMesh(aiMesh* mesh, aiMatrix4x4 transform);
	
	void traverseNode(aiNode* node, aiMatrix4x4 transform);
	void loadMaterials(const string& modelFolder);
	//void loadMeshLights();

	AssimpImporter(const AssimpImporter&) = delete;
	void operator=(const AssimpImporter&) = delete;

	ImportMode mImportMode = ImportMode::Default;
	string mFilepath;
	aiScene* mpAiScene = nullptr;
	Scene::SharedPtr mpScene;
};

KRR_NAMESPACE_END