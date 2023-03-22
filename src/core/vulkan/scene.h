#pragma once
#include <common.h>
#include <scene.h>

#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

namespace vkrhi {using namespace nvrhi;}

class VKScene {
public:
	using SharedPtr = std::shared_ptr<VKScene>;

	VKScene() = default;
	VKScene(Scene *scene) : mpScene(scene) {}
	~VKScene() = default;

protected:	
	Scene *mpScene;
	vkrhi::vulkan::DeviceHandle mDevice;

	vkrhi::BufferHandle mMaterialConstantsBuffer;
	vkrhi::BufferHandle mMeshDataBuffer;
};

KRR_NAMESPACE_END