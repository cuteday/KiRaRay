#pragma once
#include <array>
#include <common.h>
#include <scene.h>

#include "descriptor.h"
#include "textureloader.h"

#include <nvrhi/nvrhi.h>

NAMESPACE_BEGIN(krr)


namespace rs {
class MeshBuffers {
public:
	nvrhi::BufferHandle indexBuffer;
	nvrhi::BufferHandle vertexBuffer;
	DescriptorHandle indexBufferDescriptor;
	DescriptorHandle vertexBufferDescriptor;
	std::array<nvrhi::BufferRange, size_t(VertexAttribute::Count)> vertexBufferRanges;

	bool hasAttribute(VertexAttribute attr) const {
		return vertexBufferRanges[(int) attr].byteSize != 0;
	}
	nvrhi::BufferRange& getVertexBufferRange(VertexAttribute attr) {
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
	[[nodiscard]] nvrhi::ITexture *
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

// [CAUTION] structured buffer elements fulfill a 16-byte alignment.

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
	RGBA baseColor;
	RGBA specularColor;

	float IoR;
	float opacity;
	int metalRough;
	int flags;

	int baseTextureIndex;		// these indices go to the bindless textures
	int specularTextureIndex;
	int normalTextureIndex;
	int emissiveTextureIndex;
};

struct LightData {
	Vector3f direction;
	SceneLight::Type type;

	Vector3f position;
	int texture;	// for environment lights?

	RGB color;
	float scale;
};
}

class GraphicsScene {
public:
	using SharedPtr = std::shared_ptr<GraphicsScene>;

	GraphicsScene() = default;
	GraphicsScene(Scene::SharedPtr scene, nvrhi::IDevice *device,
			std::shared_ptr<DescriptorTableManager> descriptorTable = nullptr);
	~GraphicsScene() = default;

	[[nodiscard]] nvrhi::IBuffer *getMaterialBuffer() const { return mMaterialConstantsBuffer; }
	[[nodiscard]] nvrhi::IBuffer *getLightBuffer() const { return mLightDataBuffer; }
	[[nodiscard]] nvrhi::IBuffer *getInstanceBuffer() const { return mInstanceDataBuffer; }
	[[nodiscard]] nvrhi::IBuffer* getGeometryBuffer() const { return mMeshDataBuffer; }

	void update();

protected:	
	friend Scene;
	void createMeshBuffers(nvrhi::ICommandList *commandList);
	void createMaterialTextures(nvrhi::ICommandList *commandList);
	void createMaterialBuffer();	
	void createInstanceBuffer();
	void createGeometryBuffer();
	void createLightBuffer();
	
	void writeMaterialBuffer(nvrhi::ICommandList *commandList);
	void writeInstanceBuffer(nvrhi::ICommandList *commandList);
	void writeGeometryBuffer(nvrhi::ICommandList *commandList);
	void writeLightBuffer(nvrhi::ICommandList *commandList);

	size_t mLastUpdatedFrame = 0;
	std::weak_ptr<Scene> mScene{};
	nvrhi::DeviceHandle mDevice{};
	nvrhi::CommandListHandle mCommandList;
	std::shared_ptr<DescriptorTableManager> mDescriptorTable{};
	std::shared_ptr<TextureCache> mTextureLoader;

	std::vector<rs::MaterialConstants> mMaterialConstants;
	std::vector<rs::LightData> mLightData;
	std::vector<rs::MeshBuffers> mMeshBuffers;
	std::vector<rs::MaterialTextures> mMaterialTextures;
	std::vector<rs::MeshData> mMeshData;
	std::vector<rs::InstanceData> mInstanceData;
	nvrhi::BufferHandle mMaterialConstantsBuffer;
	nvrhi::BufferHandle mLightDataBuffer;
	nvrhi::BufferHandle mMeshDataBuffer;
	nvrhi::BufferHandle mInstanceDataBuffer;
};

NAMESPACE_END(krr)
