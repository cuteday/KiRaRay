#pragma once

#include "kiraray.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"

KRR_NAMESPACE_BEGIN


class Texture {
	Texture() = default;
	Texture(uint32_t* data) :pixels(data) {}

	uint32_t* pixels = nullptr;
	vec2i resolution = { 0, 0 };
};

class Mesh {
public:
	std::vector<vec3f> vertices;
	std::vector<vec3f> normals;
	std::vector<vec2f> texcoords;
	std::vector<vec3i> indices;

	Texture* texture;
	int texture_id;
};

class Scene {
public:
	using SharedPtr = std::shared_ptr<Scene>;

	Scene() = default;
	~Scene() = default;

	void createFromFile(const string filepath);
	void processMesh(aiMesh* mesh, aiMatrix4x4 transform);
	void traverse(aiNode* node, aiMatrix4x4 transform);

	//std::vector<Mesh*> meshes;
	std::vector<Mesh> meshes;
	std::vector<Texture*> textures;

//private:
	
	Assimp::Importer mImporter;
	aiScene* mScene;
};

KRR_NAMESPACE_END