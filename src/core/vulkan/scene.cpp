#include "scene.h"
#include "descriptor.h"

KRR_NAMESPACE_BEGIN

void Scene::initializeSceneVK(nvrhi::vulkan::IDevice *device, 
	std::shared_ptr<DescriptorTableManager> descriptorTable) { 
	mpSceneVK = std::make_shared<VKScene>(this, device, descriptorTable); 
	vkrhi::CommandListHandle commandList = device->createCommandList();
	commandList->open();
	mpSceneVK->writeMeshBuffers(commandList);		// bindless buffers
	mpSceneVK->writeMaterialTextures(commandList);	// bindless textures
	mpSceneVK->writeMaterialBuffer(commandList);
	mpSceneVK->writeGeometryBuffer(commandList);
	commandList->close();
	device->executeCommandList(commandList);
	device->waitForIdle();
}

void VKScene::writeMeshBuffers(vkrhi::ICommandList *commandList) {
	auto appendBufferRange = [](vkrhi::BufferRange &range, size_t size,
								uint64_t &currentBufferSize) {
		range.byteOffset = currentBufferSize;
		range.byteSize	 = size;
		currentBufferSize += size;
	};
	mMeshBuffers.clear();
	for (const auto &mesh : mpScene->getMeshes()) {
		mMeshBuffers.push_back(rs::MeshBuffers());
		rs::MeshBuffers &buffers = mMeshBuffers.back();
		
		/* Create and write index buffer. */
		vkrhi::BufferDesc bufferDesc;
		bufferDesc.isIndexBuffer	 = true;
		bufferDesc.byteSize			 = mesh->indices.size() * sizeof(Vector3i);
		bufferDesc.debugName		 = "IndexBuffer";
		bufferDesc.canHaveTypedViews = true;
		bufferDesc.canHaveRawViews	 = true;
		bufferDesc.format			 = nvrhi::Format::R32_UINT;
		buffers.indexBuffer			 = mDevice->createBuffer(bufferDesc);
	
		commandList->beginTrackingBufferState(buffers.indexBuffer,
											  nvrhi::ResourceStates::Common);
		commandList->writeBuffer(buffers.indexBuffer, mesh->indices.data(),
								 bufferDesc.byteSize);
		commandList->setPermanentBufferState(buffers.indexBuffer,
			vkrhi::ResourceStates::IndexBuffer | vkrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();
		
		/* Create vertex attribute buffer. */
		bufferDesc = vkrhi::BufferDesc();
		bufferDesc.isVertexBuffer	 = true;
		bufferDesc.byteSize			 = 0;
		bufferDesc.debugName		 = "VertexBuffer";
		bufferDesc.canHaveTypedViews = true;
		bufferDesc.canHaveRawViews	 = true;
		
		if (!mesh->positions.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Position), 
				mesh->positions.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		if (!mesh->normals.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Normal),
				mesh->normals.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		if (!mesh->texcoords.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Texcoord),
				mesh->texcoords.size() * sizeof(Vector2f), bufferDesc.byteSize);
		}
		if (!mesh->tangents.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Tangent),
				mesh->tangents.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		
		buffers.vertexBuffer = mDevice->createBuffer(bufferDesc);
		
		commandList->beginTrackingBufferState(buffers.vertexBuffer,
											  vkrhi::ResourceStates::Common);
		if (!mesh->positions.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Position);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh->positions.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh->normals.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Normal);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh->normals.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh->texcoords.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Texcoord);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh->texcoords.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh->tangents.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Tangent);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh->tangents.data(), range.byteSize, range.byteOffset);
		}

		commandList->setPermanentBufferState(buffers.vertexBuffer, 
			vkrhi::ResourceStates::VertexBuffer | vkrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();

		if (mDescriptorTable) {
			/* Create descriptors for bindless (vertex) buffers and textures. */
			buffers.indexBufferDescriptor = mDescriptorTable->CreateDescriptorHandle(
				vkrhi::BindingSetItem::RawBuffer_SRV(0, buffers.indexBuffer));
			buffers.vertexBufferDescriptor = mDescriptorTable->CreateDescriptorHandle(
				vkrhi::BindingSetItem::RawBuffer_SRV(0, buffers.vertexBuffer));
		}
	}
}

void VKScene::writeMaterialTextures(vkrhi::ICommandList* commandList) {
	mMaterialTextures.clear();
	if (!mTextureLoader) mTextureLoader =
			std::make_shared<TextureCache>(mDevice, mDescriptorTable);
	for (auto material : mpScene->getMaterials()) {
		mMaterialTextures.push_back(rs::MaterialTextures());
		auto &textures = mMaterialTextures.back();
		for (int type = 0; type < (int) Material::TextureType::Count; type++) {
			if (material->hasTexture((Material::TextureType) type) &&
				material->mTextures[type]->getImage()) {
				Log(Debug, "Loading texture slot %d for material %s", type, material->getName());
				// Upload texture to vulkan device...
				Image::SharedPtr image = material->mTextures[type]->getImage();
				auto loadedTexture = mTextureLoader->LoadTextureFromImage(image, commandList);
				textures.textures[type] = loadedTexture;
			}
		}
	}
}

void VKScene::writeMaterialBuffer(vkrhi::ICommandList *commandList) {
	/* Fill material constants buffer on host. */
	auto &materials = mpScene->getMaterials();
	for (int i = 0; i < materials.size(); i++) {
		const auto &material = materials[i];
		rs::MaterialConstants materialConstants;
		materialConstants.baseColor = material->mMaterialParams.diffuse;
		materialConstants.specularColor = material->mMaterialParams.specular;
		materialConstants.IoR			= material->mMaterialParams.IoR;
		materialConstants.opacity		= material->mMaterialParams.specularTransmission;
		materialConstants.metalRough = material->mShadingModel == Material::ShadingModel::MetallicRoughness;
		
		const auto &textures = mMaterialTextures[i];
		materialConstants.baseTextureIndex =
			textures.getDescriptor(Material::TextureType::Diffuse);
		materialConstants.specularTextureIndex =
			textures.getDescriptor(Material::TextureType::Specular);
		materialConstants.normalTextureIndex =
			textures.getDescriptor(Material::TextureType::Normal);
		materialConstants.emissiveTextureIndex =
			textures.getDescriptor(Material::TextureType::Emissive);
		mMaterialConstants.push_back(materialConstants);
	}
	
	/* Create and write material constants buffer. */
	mMaterialConstantsBuffer = nullptr;
	vkrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize = sizeof(rs::MaterialConstants) * materials.size();
	bufferDesc.debugName		= "BindlessMaterials";
	bufferDesc.structStride		= sizeof(rs::MaterialConstants);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= vkrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mMaterialConstantsBuffer	= mDevice->createBuffer(bufferDesc);
	
	commandList->writeBuffer(
		mMaterialConstantsBuffer, mMaterialConstants.data(),
		mMaterialConstants.size() * sizeof(rs::MaterialConstants), 0);
}

void VKScene::writeGeometryBuffer(vkrhi::ICommandList *commandList) {
	/* Fill mesh data buffer on host. */
	/* Normally, a instance is from a mesh, which may contain several geometries.
		In kiraray, we simply ignore this (i.e. the concept of geometry and instances). */
	auto meshes = mpScene->getMeshes();
	for (int i = 0; i < meshes.size(); i++) {
		const auto &mesh = meshes[i];
		rs::MeshData meshData;
		meshData.materialIndex = mesh->materialId;
		meshData.numIndices	   = mesh->indices.size();
		meshData.numVertices   = mesh->positions.size();
		// the descriptorHandle.Get will return -1 if invalid.
		meshData.indexBufferIndex = mMeshBuffers[i].indexBufferDescriptor.Get();
		meshData.vertexBufferIndex = mMeshBuffers[i].vertexBufferDescriptor.Get();
		meshData.indexOffset = 0;
		meshData.positionOffset = mMeshBuffers[i].hasAttribute(VertexAttribute::Position)
			? mMeshBuffers[i].getVertexBufferRange(VertexAttribute::Position).byteOffset : ~0u;
		meshData.normalOffset = mMeshBuffers[i].hasAttribute(VertexAttribute::Normal)
			? mMeshBuffers[i].getVertexBufferRange(VertexAttribute::Normal).byteOffset : ~0u;
		meshData.texCoordOffset = mMeshBuffers[i].hasAttribute(VertexAttribute::Texcoord)
			? mMeshBuffers[i].getVertexBufferRange(VertexAttribute::Texcoord).byteOffset : ~0u;
		meshData.tangentOffset = mMeshBuffers[i].hasAttribute(VertexAttribute::Tangent)
			? mMeshBuffers[i].getVertexBufferRange(VertexAttribute::Tangent).byteOffset : ~0u;
		
		mMeshData.push_back(meshData);
	}
	/* Create and write mesh data buffer. */
	vkrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize			= sizeof(rs::MeshData) * mMeshData.size();
	bufferDesc.debugName		= "BindlessMesh";
	bufferDesc.structStride		= sizeof(rs::MeshData);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= vkrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mMeshDataBuffer				= mDevice->createBuffer(bufferDesc);
	
	commandList->writeBuffer(mMeshDataBuffer, mMeshData.data(), bufferDesc.byteSize, 0);
}

KRR_NAMESPACE_END