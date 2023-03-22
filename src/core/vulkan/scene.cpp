#include "scene.h"

KRR_NAMESPACE_BEGIN

void Scene::initializeSceneVK() { 
	mpSceneVK = std::make_shared<VKScene>(this); 
	// create mesh buffers
	// create mesh data and material constants buffer
}

KRR_NAMESPACE_END