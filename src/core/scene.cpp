#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"

#include "window.h"
#include "scene.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

Scene::Scene() {
    // other components
    mpEnvLight = EnvLight::SharedPtr(new EnvLight());
    mpCamera = Camera::SharedPtr(new Camera());
    mpCameraController = OrbitCameraController::SharedPtr(new OrbitCameraController(mpCamera));
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