#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"

#include "window.h"
#include "scene.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

namespace {
    vec3f aiCast(aiVector3D v) {
        return { v.x, v.y, v.z };
    }

    vec2f aiCast(aiVector2D v) {
        return { v.x, v.y };
    }
}

Scene::Scene() {
    // other components
    mpEnvLight = EnvLight::SharedPtr(new EnvLight());
    mpCamera = Camera::SharedPtr(new Camera());
    mpCameraController = OrbitCameraController::SharedPtr(new OrbitCameraController(mpCamera));
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

    Assimp::Importer importer;
    mScene = (aiScene*)importer.ReadFile(filepath, postProcessSteps);
    if (!mScene) logFatal("Assimp::load model failed");

    traverseNode(mScene->mRootNode, aiMatrix4x4());

    logDebug("Total imported meshes: " + std::to_string(meshes.size()));

    Assimp::DefaultLogger::kill();
}

void Scene::processMesh(aiMesh* pAiMesh, aiMatrix4x4 transform)
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

    meshes.push_back(mesh);
}


void Scene::traverseNode(aiNode* node, aiMatrix4x4 transform)
{
    transform = transform * node->mTransformation;

    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = mScene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, transform);
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        traverseNode(node->mChildren[i], transform);
    }
}

void Scene::createUnitCube()
{
    Mesh cube;
    cube.vertices.push_back(vec3f(0.f, 0.f, 0.f));
    cube.vertices.push_back(vec3f(1.f, 0.f, 0.f));
    cube.vertices.push_back(vec3f(0.f, 1.f, 0.f));
    cube.vertices.push_back(vec3f(1.f, 1.f, 0.f));
    cube.vertices.push_back(vec3f(0.f, 0.f, 1.f));
    cube.vertices.push_back(vec3f(1.f, 0.f, 1.f));
    cube.vertices.push_back(vec3f(0.f, 1.f, 1.f));
    cube.vertices.push_back(vec3f(1.f, 1.f, 1.f));

    int indices[] = { 0,1,3, 2,3,0,
                 5,7,6, 5,6,4,
                 0,4,5, 0,5,1,
                 2,3,7, 2,7,6,
                 1,5,7, 1,7,3,
                 4,0,2, 4,2,6
    };
    for (int i = 0; i < 12; i++)
        cube.indices.push_back(
          vec3i(indices[3 * i + 0],
                indices[3 * i + 1],
                indices[3 * i + 2]));
    meshes.push_back(cube);
}

void Scene::update()
{
    mpCameraController->update();   // camera.update() is called within.
}

void Scene::renderUI() {
    
    ui::Text("Hello from scene!");
    if (mpCamera && ui::CollapsingHeader("Camera")) {
        mpCamera->renderUI();
    }
}

bool Scene::onMouseEvent(const MouseEvent& mouseEvent)
{
    if(mpCameraController && mpCameraController->onMouseEvent(mouseEvent))
        return true;
    return false;
}

bool Scene::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if(mpCameraController && mpCameraController->onKeyEvent(keyEvent))
        return true;
    return false;
}


KRR_NAMESPACE_END