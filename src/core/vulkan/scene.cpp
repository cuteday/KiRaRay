#include "scene.h"
#include "descriptor.h"

KRR_NAMESPACE_BEGIN

void Scene::initializeSceneVK(nvrhi::vulkan::IDevice *device, 
	DescriptorTableManager *descriptorTable) { 
	mpSceneVK = std::make_shared<VKScene>(this, device); 
	vkrhi::CommandListHandle commandList = device->createCommandList();
	commandList->open();
	mpSceneVK->writeMeshBuffers(commandList);
	mpSceneVK->writeMaterialBuffer(commandList);
	mpSceneVK->writeGeometryBuffer(commandList);
	commandList->close();
	device->executeCommandList(commandList);
	if (descriptorTable) mpSceneVK->writeDescriptorTable(descriptorTable);
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
	for (const auto &mesh : mpScene->meshes) {
		mMeshBuffers.push_back(rs::MeshBuffers());
		rs::MeshBuffers &buffers = mMeshBuffers.back();
		
		/* Create and write index buffer. */
		vkrhi::BufferDesc bufferDesc;
		bufferDesc.isIndexBuffer	 = true;
		bufferDesc.byteSize			 = mesh.indices.size() * sizeof(Vector3i);
		bufferDesc.debugName		 = "IndexBuffer";
		bufferDesc.canHaveTypedViews = true;
		bufferDesc.canHaveRawViews	 = true;
		bufferDesc.format			 = nvrhi::Format::R32_UINT;
		buffers.indexBuffer			 = mDevice->createBuffer(bufferDesc);
	
		commandList->beginTrackingBufferState(buffers.indexBuffer,
											  nvrhi::ResourceStates::Common);
		commandList->writeBuffer(buffers.indexBuffer, mesh.indices.data(),
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
		
		if (!mesh.positions.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Position), 
				mesh.positions.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		if (!mesh.normals.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Normal),
				mesh.normals.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		if (!mesh.texcoords.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Texcoord),
				mesh.texcoords.size() * sizeof(Vector2f), bufferDesc.byteSize);
		}
		if (!mesh.tangents.empty()) {
			appendBufferRange(buffers.getVertexBufferRange(VertexAttribute::Tangent),
				mesh.tangents.size() * sizeof(Vector3f), bufferDesc.byteSize);
		}
		
		buffers.vertexBuffer = mDevice->createBuffer(bufferDesc);
		
		commandList->beginTrackingBufferState(buffers.vertexBuffer,
											  vkrhi::ResourceStates::Common);
		if (!mesh.positions.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Position);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh.positions.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh.normals.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Normal);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh.normals.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh.texcoords.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Texcoord);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh.texcoords.data(), range.byteSize, range.byteOffset);
		}
		if (!mesh.tangents.empty()) {
			const auto &range = buffers.getVertexBufferRange(VertexAttribute::Tangent);
			commandList->writeBuffer(buffers.vertexBuffer,
									 mesh.tangents.data(), range.byteSize, range.byteOffset);
		}

		commandList->setPermanentBufferState(buffers.vertexBuffer, 
			vkrhi::ResourceStates::VertexBuffer | vkrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();
	}
}

void VKScene::writeMaterialBuffer(vkrhi::ICommandList *commandList) {
	/* Fill material constants buffer on host. */
	for (const auto& material : mpScene->materials) {
		rs::MaterialConstants materialConstants;
		materialConstants.baseColor = material.mMaterialParams.diffuse;
		materialConstants.specularColor = material.mMaterialParams.specular;
		materialConstants.IoR			= material.mMaterialParams.IoR;
		materialConstants.opacity = material.mMaterialParams.specularTransmission;
		materialConstants.metalRough = material.mShadingModel == Material::ShadingModel::MetallicRoughness;
		mMaterialConstants.push_back(materialConstants);
	}
	
	/* Create and write material constants buffer. */
	mMaterialConstantsBuffer = nullptr;
	vkrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize = sizeof(rs::MaterialConstants) * mpScene->materials.size();
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
	for (int i = 0; i < mpScene->meshes.size(); i++) {
		const auto &mesh = mpScene->meshes[i];
		rs::MeshData meshData;
		meshData.materialIndex = mesh.materialId;
		meshData.numIndices	   = mesh.indices.size();
		meshData.numVertices   = mesh.positions.size();
		// the descriptorHandle.Get will return -1 if invalid.
		meshData.indexBufferIndex = mMeshBuffers[i].indexBufferDescriptor.Get();
		meshData.vertexBufferIndex = mMeshBuffers[i].vertexBufferDescriptor.Get();
		
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

void VKScene::writeDescriptorTable(DescriptorTableManager *descriptorTable) {
	for (rs::MeshBuffers &buffers : mMeshBuffers) {
		buffers.indexBufferDescriptor = descriptorTable->CreateDescriptorHandle(
			vkrhi::BindingSetItem::RawBuffer_SRV(0, buffers.indexBuffer));
		buffers.vertexBufferDescriptor = descriptorTable->CreateDescriptorHandle(
				vkrhi::BindingSetItem::RawBuffer_SRV(0, buffers.vertexBuffer));
	}
}


KRR_NAMESPACE_END