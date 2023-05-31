#pragma once
#include <array>
#include <common.h>
#include <scene.h>

#include "descriptor.h"
#include "textureloader.h"

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
	std::array<vkrhi::BufferRange, size_t(VertexAttribute::Count)> vertexBufferRanges;

	bool hasAttribute(VertexAttribute attr) const {
		return vertexBufferRanges[(int) attr].byteSize != 0;
	}
	vkrhi::BufferRange& getVertexBufferRange(VertexAttribute attr) {
		return vertexBufferRanges[(int) attr];
	}
};

class MaterialTextures {
public:
	std::array<std::shared_ptr<LoadedTexture>, 
		size_t(Material::TextureType::Count)> textures;

	bool hasTexture(Material::TextureType textureType) const {
		return textures[(size_t) textureType].get();
	}
	[[nodiscard]] vkrhi::ITexture *
		getTexture(Material::TextureType textureType) const {
		if (!hasTexture(textureType)) {
			Log(Error, "Attempt to get texture that a material do not pocess.");
			return nullptr;
		}
		return textures[(size_t) textureType]->texture;
	}
	[[nodiscard]] int getDescriptor(Material::TextureType textureType) const {
		if (!hasTexture(textureType)) return -1;
		return textures[(size_t) textureType]->bindlessDescriptor.Get();
	}
};

// structured buffer elements fulfill a 16-byte alignment.

struct MeshData {
	uint numIndices;
	uint numVertices;
	int indexBufferIndex;		// these indices go to the bindless buffers
	int vertexBufferIndex;		// vertex data are placed within one buffer

	uint positionOffset;		
	uint normalOffset;
	uint texCoordOffset;
	uint tangentOffset;		

	uint indexOffset;
	uint materialIndex;			// this indexes the material constants buffer
	Vector2i padding;
};

struct InstanceData {
	uint meshIndex;
	Vector3i padding;

	Matrix4f transform;
};

struct MaterialConstants {
	Color4f baseColor;
	Color4f specularColor;

	float IoR;
	float opacity;
	int metalRough;
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
	VKScene(Scene::SharedPtr scene, vkrhi::vulkan::IDevice* device, 
		std::shared_ptr<DescriptorTableManager> descriptorTable = nullptr) : 
		mScene(scene), mDevice(device), mDescriptorTable(descriptorTable) {}
	~VKScene() = default;

	[[nodiscard]] vkrhi::IBuffer* getMaterialBuffer() const { return mMaterialConstantsBuffer; }
	[[nodiscard]] vkrhi::IBuffer *getInstanceBuffer() const { return mInstanceDataBuffer; }
	[[nodiscard]] vkrhi::IBuffer* getGeometryBuffer() const { return mMeshDataBuffer; }

protected:	
	friend Scene;
	void writeMeshBuffers(vkrhi::ICommandList *commandList);
	void writeMaterialTextures(vkrhi::ICommandList *commandList);
	
	void writeMaterialBuffer(vkrhi::ICommandList *commandList);
	void writeInstanceBuffer(vkrhi::ICommandList *commandList);
	void writeGeometryBuffer(vkrhi::ICommandList *commandList);

	Scene::SharedPtr mScene{};
	vkrhi::vulkan::DeviceHandle mDevice{};
	std::shared_ptr<DescriptorTableManager> mDescriptorTable{};
	std::shared_ptr<TextureCache> mTextureLoader;

	std::vector<rs::MaterialConstants> mMaterialConstants;
	std::vector<rs::MeshBuffers> mMeshBuffers;
	std::vector<rs::MaterialTextures> mMaterialTextures;
	std::vector<rs::MeshData> mMeshData;
	std::vector<rs::InstanceData> mInstanceData;
	vkrhi::BufferHandle mMaterialConstantsBuffer;
	vkrhi::BufferHandle mMeshDataBuffer;
	vkrhi::BufferHandle mInstanceDataBuffer;
};

KRR_NAMESPACE_END