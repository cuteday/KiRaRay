#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/pbrmaterial.h"

#include "scene/importer.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN

namespace assimp {
    vec3f aiCast(const aiColor3D& ai) { return vec3f(ai.r, ai.g, ai.b); }
    vec3f aiCast(const aiVector3D& val) { return vec3f(val.x, val.y, val.z); }
    quat aiCast(const aiQuaternion& q) { return quat(q.w, q.x, q.y, q.z); }
}

using namespace krr::assimp;
bool AssimpImporter::import(const string& filepath, const Scene::SharedPtr pScene) {
    Assimp::DefaultLogger::create("", Assimp::Logger::NORMAL, aiDefaultLogStream_STDOUT);
    Assimp::DefaultLogger::get()->info("KRR::Assimp::DefaultLogger initialized!");

    unsigned int postProcessSteps = 0
        | aiProcess_CalcTangentSpace
        //| aiProcess_JoinIdenticalVertices
        //| aiProcess_MakeLeftHanded
        | aiProcess_Triangulate
        //| aiProcess_RemoveComponent
        //| aiProcess_GenNormals
        | aiProcess_GenSmoothNormals
        | aiProcess_RemoveRedundantMaterials
        //| aiProcess_FixInfacingNormals
        | aiProcess_SortByPType
        | aiProcess_GenUVCoords
        | aiProcess_TransformUVCoords
        //| aiProcess_FlipUVs
        //| aiProcess_FlipWindingOrder
        ;

    Assimp::Importer importer;
    mpScene = pScene;
    mpAiScene = (aiScene*)importer.ReadFile(filepath, postProcessSteps);
    if (!mpAiScene) logFatal("Assimp::load model failed");

    // load materials
    for (int i = 0; i < mpAiScene->mNumMaterials; i++) {
        aiMaterial* material = mpAiScene->mMaterials[i];
        logInfo("Found material: " + string(material->GetName().C_Str()));
    }

    traverseNode(mpAiScene->mRootNode, aiMatrix4x4());

    logDebug("Total imported meshes: " + std::to_string(mpScene->meshes.size()));

    Assimp::DefaultLogger::kill();
    
    return true;
}

void AssimpImporter::processMesh(aiMesh* pAiMesh, aiMatrix4x4 transform)
{
    Mesh mesh;

    for (uint i = 0; i < pAiMesh->mNumVertices; i++) {
        vec3f vertex = aiCast(pAiMesh->mVertices[i]);
        mesh.vertices.push_back(vertex);

        assert(pAiMesh->HasNormals());
        vec3f normal = aiCast(pAiMesh->mNormals[i]);
        mesh.normals.push_back(normal);

        if (pAiMesh->mTextureCoords[0]) {
            vec3f texcoord = aiCast(pAiMesh->mTextureCoords[0][i]);
            mesh.texcoords.push_back({ texcoord.x, texcoord.y });
        }

    }

    for (uint i = 0; i < pAiMesh->mNumFaces; i++) {
        aiFace face = pAiMesh->mFaces[i];
        assert(face.mNumIndices == 3);
        vec3i indices = { (int)face.mIndices[0], (int)face.mIndices[1], (int)face.mIndices[2] };

        mesh.indices.push_back(indices);
    }

    mpScene->meshes.push_back(mesh);
}

void AssimpImporter::traverseNode(aiNode* node, aiMatrix4x4 transform)
{
    transform = transform * node->mTransformation;

    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = mpAiScene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, transform);
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        traverseNode(node->mChildren[i], transform);
    }
}

KRR_NAMESPACE_END