#pragma once
#include <array>
#include <common.h>
#include <scene.h>
#include "descriptor.h"

#include <nvrhi/vulkan.h>

KRR_NAMESPACE_BEGIN

namespace vkrhi {using namespace nvrhi;}

namespace rs {
class MeshBuffers {
public:
	vkrhi::BufferHandle indexBuffer; 
	vkrhi::BufferHandle vertexBuffer;
	DescriptorHandle indexBufferDescriptor;
	DescriptorHandle vertexBufferDescriptor;
	std::array<vkrhi::BufferRange, size_t(VertexAttributeType::Count)> vertexBufferRanges;

	bool hasAttribute(VertexAttributeType attr) const {
		return vertexBufferRanges[(int) attr].byteSize != 0;
	}
	vkrhi::BufferRange& getVertexBufferRange(VertexAttributeType attr) {
		return vertexBufferRanges[(int) attr];
	}
};

struct MeshData {
	uint numIndices;
	uint numVertices;
	int indexBufferIndex;		// these indices go to the bindless buffers
	int vertexBufferIndex;

	uint positionOffset;
	uint normalOffset;
	uint texCoordOffset;
	uint tangentOffset;		

	uint bitangentOffset;
	uint materialIndex;			// this indexes the material constants buffer
	Vector2i padding;
};

struct MaterialConstants {
	Color3f baseColor;
	float roughness;

	Color3f specularColor;
	float opacity;

	Color3f emissiveColor;
	int flags;

	int baseTextureIndex;		// these indices go to the bindless textures
	int specularTextureIndex;
	int normalTextureIndex;
	int emissiveTextureIndex;
};
}

class VKScene {
public:
	using SharedPtr = std::shared_ptr<VKScene>;

	VKScene() = default;
	VKScene(Scene *scene) : mpScene(scene) {}
	~VKScene() = default;

protected:	
	Scene *mpScene;
	vkrhi::vulkan::DeviceHandle mDevice;

	std::vector<rs::MaterialConstants> mMaterialConstants;
	std::vector<rs::MeshBuffers> mMeshBuffers;
	std::vector<rs::MeshData> mMeshData;
	vkrhi::BufferHandle mMaterialConstantsBuffer;
	vkrhi::BufferHandle mMeshDataBuffer;
};

KRR_NAMESPACE_END