#pragma once
#include <map>

#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "scene.h"
#include "texture.h"

KRR_NAMESPACE_BEGIN

namespace importer {

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

class AssimpImporter {
public:
	enum class ImportMode {
		Default = 0,
		OBJ,
		GLTF2,
	};
	AssimpImporter() = default;
	bool import(const string &filepath, Scene::SharedPtr pScene);

private:
	AssimpImporter(const AssimpImporter &) = delete;
	void operator=(const AssimpImporter &) = delete;

	void processMesh(aiMesh *mesh, aiMatrix4x4 transform);
	void traverseNode(aiNode *node, aiMatrix4x4 transform);
	void loadMaterials(const string &modelFolder);

	ImportMode mImportMode = ImportMode::Default;
	string mFilepath;
	aiScene *mpAiScene = nullptr;
	Scene::SharedPtr mpScene;
};

class PbrtImporter {
public:
	PbrtImporter() = default;
	bool import(const string &filepath, Scene::SharedPtr pScene);

private:
	PbrtImporter(const PbrtImporter &) = delete;
	void operator=(const PbrtImporter &) = delete;

	string resolve(string path);

	string mFilepath;
	string mBasepath;
	Scene::SharedPtr mpScene;
};

inline bool loadScene(const fs::path filepath, Scene::SharedPtr pScene) {
	bool success{};
	string format = filepath.extension().string();
	if (format == ".obj" || format == ".gltf" || format == ".glb" || format == ".fbx") {
		success = AssimpImporter().import(filepath.string(), pScene);
	} else if (format == ".pbrt") {
		success = PbrtImporter().import(filepath.string(), pScene);
	} else {
		Log(Fatal, "Unsupported file format: %s...", format.c_str());
		return false;
	}
	if (!success)
		Log(Fatal, "Failed to load scene file from %ls...", filepath.c_str());
	return false;
}

} // namespace importer

KRR_NAMESPACE_END