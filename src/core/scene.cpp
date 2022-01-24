#include "scene.h"

#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"

KRR_NAMESPACE_BEGIN

namespace {
    vec3f aiCast(aiVector3D v) {
        return { v.x, v.y, v.z };
    }

    vec2f aiCast(aiVector2D v) {
        return { v.x, v.y };
    }
}

void Scene::createFromFile(const string filepath)
{
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

    mScene = (aiScene*)mImporter.ReadFile(filepath, postProcessSteps);
    if (!mScene) throw std::runtime_error("Assimp::load model failed");

    traverse(mScene->mRootNode, aiMatrix4x4());

    std::stringstream info;
    info << "Total imported meshes: " << meshes.size();

    Assimp::DefaultLogger::get()->info(info.str());
    Assimp::DefaultLogger::kill();
}

void Scene::processMesh(aiMesh* pAiMesh, aiMatrix4x4 transform)
{
    Mesh mesh;

    for (uint i = 0; i < pAiMesh->mNumVertices; i++) {
        vec3f vertex = aiCast(pAiMesh->mVertices[i]);
        mesh.vertex.push_back(vertex);

        assert(pAiMesh->HasNormals());
        vec3f normal = aiCast(pAiMesh->mNormals[i]);
        mesh.normal.push_back(normal);

        if (pAiMesh->mTextureCoords[0]) {
            vec3f texcoord = aiCast(pAiMesh->mTextureCoords[0][i]);
            mesh.texcoord.push_back({ texcoord.x, texcoord.y });
        }

    }

    for (uint i = 0; i < pAiMesh->mNumFaces; i++) {
        aiFace face = pAiMesh->mFaces[i];
        assert(face.mNumIndices == 3);
        vec3i indices = { (int)face.mIndices[0], (int)face.mIndices[1], (int)face.mIndices[2] };
        
        mesh.index.push_back(indices);
    }

    meshes.push_back(mesh);
}


void Scene::traverse(aiNode* node, aiMatrix4x4 transform)
{
    transform = transform * node->mTransformation;

    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = mScene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, transform);
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        traverse(node->mChildren[i], transform);
    }
}

KRR_NAMESPACE_END