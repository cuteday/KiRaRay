#include "scene.h"
#include "descriptor.h"

KRR_NAMESPACE_BEGIN

void Scene::initializeSceneVK(
	vkrhi::ICommandList *commandList,
	DescriptorTableManager *descriptorTable) { 
	mpSceneVK = std::make_shared<VKScene>(this); 

	mpSceneVK->writeMeshBuffers(commandList);
	mpSceneVK->writeMaterialBuffer(commandList);
	mpSceneVK->writeGeometryBuffer(commandList);
}

void VKScene::writeMeshBuffers(vkrhi::ICommandList *commandList) const {

}

void VKScene::writeMaterialBuffer(vkrhi::ICommandList *commandList) const {

}

void VKScene::writeGeometryBuffer(vkrhi::ICommandList *commandList) const {

}


KRR_NAMESPACE_END