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