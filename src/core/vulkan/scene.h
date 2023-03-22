#pragma once
#include <common.h>

#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

class VKScene {
public:
	using SharedPtr = std::shared_ptr<VKScene>;

	VKScene() = default;
	VKScene(Scene *scene) : mpScene(scene) {}
	~VKScene() = default;

private:
	Scene *mpScene;

	vkrhi::BufferHandle mMaterialConstantsBuffer;
	vkrhi::BufferHandle mMeshDataBuffer;
};

KRR_NAMESPACE_END