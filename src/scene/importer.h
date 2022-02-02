#pragma once

#include "assimp/scene.h"
#include "assimp/Importer.hpp"

#include "common.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class AssimpImporter {
public:
    AssimpImporter() = default;
    bool import(const string& filepath, Scene::SharedPtr pScene);

private:

    void createAllMaterials() {}
    void processMesh(aiMesh* mesh, aiMatrix4x4 transform);
    void traverseNode(aiNode* node, aiMatrix4x4 transform);

    AssimpImporter(const AssimpImporter&) = delete;
    void operator=(const AssimpImporter&) = delete;

    aiScene* mpAiScene = nullptr;
    Scene::SharedPtr mpScene;
};

KRR_NAMESPACE_END