#include "scene.h"
#include "descriptor.h"

NAMESPACE_BEGIN(krr)

void Scene::initializeGraphicsScene(nvrhi::IDevice *device,
	std::shared_ptr<DescriptorTableManager> descriptorTable) { 
	if (!mGraph) Log(Fatal, "Scene graph must be initialized.");
	if (mGraphicsScene) {
		Log(Warning, "The graphics scene data has been initialized once before."
					 "I'm assuming you do not want to reinitialize it?");
		return;
	}
	mGraph->update(0); // must be done before preparing device data.
	mGraphicsScene = std::make_shared<GraphicsScene>(shared_from_this(), device, descriptorTable);
	nvrhi::CommandListHandle commandList = device->createCommandList();
	commandList->open();
	mGraphicsScene->createMeshBuffers(commandList);		// bindless buffers
	mGraphicsScene->createMaterialTextures(commandList);	// bindless textures
	mGraphicsScene->createInstanceBuffer();
	mGraphicsScene->createMaterialBuffer();
	mGraphicsScene->createGeometryBuffer();
	mGraphicsScene->createLightBuffer();
	mGraphicsScene->writeInstanceBuffer(commandList);
	mGraphicsScene->writeMaterialBuffer(commandList);
	mGraphicsScene->writeGeometryBuffer(commandList);
	mGraphicsScene->writeLightBuffer(commandList);
	commandList->close();
	device->executeCommandList(commandList);
	device->waitForIdle();
}

GraphicsScene::GraphicsScene(Scene::SharedPtr scene, nvrhi::IDevice *device,
	std::shared_ptr<DescriptorTableManager> descriptorTable) :	
	mScene(scene), mDevice(device), mDescriptorTable(descriptorTable) {
	mCommandList = mDevice->createCommandList();
}

void GraphicsScene::createMeshBuffers(nvrhi::ICommandList *commandList) {
	auto appendBufferRange = [](nvrhi::BufferRange &range, size_t size,
								uint64_t &currentBufferSize) {
		range.byteOffset = currentBufferSize;
		range.byteSize	 = size;
		currentBufferSize += size;
	};
	mMeshBuffers.clear();
	for (const auto &mesh : mScene.lock()->getMeshes()) {
		mMeshBuffers.push_back(rs::MeshBuffers());
		rs::MeshBuffers &buffers = mMeshBuffers.back();
		
		/* Create and write index buffer. */
		nvrhi::BufferDesc bufferDesc;
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
			nvrhi::ResourceStates::IndexBuffer | nvrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();
		
		/* Create vertex attribute buffer. */
		bufferDesc = nvrhi::BufferDesc();
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
											  nvrhi::ResourceStates::Common);
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
			nvrhi::ResourceStates::VertexBuffer | nvrhi::ResourceStates::ShaderResource);
		commandList->commitBarriers();

		if (mDescriptorTable) {
			/* Create descriptors for bindless (vertex) buffers and textures. */
			buffers.indexBufferDescriptor = mDescriptorTable->CreateDescriptorHandle(
				nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers.indexBuffer));
			buffers.vertexBufferDescriptor = mDescriptorTable->CreateDescriptorHandle(
				nvrhi::BindingSetItem::RawBuffer_SRV(0, buffers.vertexBuffer));
		}
	}
}

void GraphicsScene::createMaterialTextures(nvrhi::ICommandList* commandList) {
	mMaterialTextures.clear();
	if (!mTextureLoader) mTextureLoader =
			std::make_shared<TextureCache>(mDevice, mDescriptorTable);
	for (auto material : mScene.lock()->getMaterials()) {
		mMaterialTextures.push_back(rs::MaterialTextures());
		auto &textures = mMaterialTextures.back();
		for (int type = 0; type < (int) Material::TextureType::Count; type++) {
			if (material->hasTexture((Material::TextureType) type) &&
				material->mTextures[type]->getImage()) {
				Log(Debug, "Loading texture slot %d for material %s", type, material->getName());
				// Upload texture to graphics device...
				Image::SharedPtr image = material->mTextures[type]->getImage();
				auto loadedTexture = mTextureLoader->LoadTextureFromImage(image, commandList);
				textures.textures[type] = loadedTexture;
			}
		}
	}
}

void GraphicsScene::createMaterialBuffer() {
	/* Create and write material constants buffer. */
	mMaterialConstantsBuffer = nullptr;
	nvrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize	 = sizeof(rs::MaterialConstants) * mScene.lock()->getMaterials().size();
	bufferDesc.debugName		= "BindlessMaterials";
	bufferDesc.structStride		= sizeof(rs::MaterialConstants);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= nvrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mMaterialConstantsBuffer	= mDevice->createBuffer(bufferDesc);
}

void GraphicsScene::createInstanceBuffer() {
	/* Create and write instance data buffer. */
	nvrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize	 = sizeof(rs::InstanceData) * mScene.lock()->getMeshInstances().size();
	bufferDesc.debugName		= "BindlessInstance";
	bufferDesc.structStride		= sizeof(rs::InstanceData);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= nvrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mInstanceDataBuffer			= mDevice->createBuffer(bufferDesc);
}

void GraphicsScene::createGeometryBuffer() {
	/* Create and write mesh data buffer. */
	nvrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize			= sizeof(rs::MeshData) * mScene.lock()->getMeshes().size();
	bufferDesc.debugName		= "BindlessMesh";
	bufferDesc.structStride		= sizeof(rs::MeshData);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= nvrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mMeshDataBuffer				= mDevice->createBuffer(bufferDesc);
}

void GraphicsScene::createLightBuffer() {
	nvrhi::BufferDesc bufferDesc;
	bufferDesc.byteSize			= sizeof(rs::LightData) * mScene.lock()->getLights().size();
	bufferDesc.debugName		= "BindlessLights";
	bufferDesc.structStride		= sizeof(rs::LightData);
	bufferDesc.canHaveRawViews	= true;
	bufferDesc.canHaveUAVs		= true;
	bufferDesc.initialState		= nvrhi::ResourceStates::ShaderResource;
	bufferDesc.keepInitialState = true;
	mLightDataBuffer			= mDevice->createBuffer(bufferDesc);
}

void GraphicsScene::writeMaterialBuffer(nvrhi::ICommandList *commandList) {
	/* Fill material constants buffer on host. */
	auto &materials = mScene.lock()->getMaterials();
	for (int i = 0; i < materials.size(); i++) {
		const auto &material = materials[i];
		rs::MaterialConstants materialConstants;
		materialConstants.baseColor		= material->mMaterialParams.diffuse;
		materialConstants.specularColor = material->mMaterialParams.specular;
		materialConstants.IoR			= material->mMaterialParams.IoR;
		materialConstants.opacity		= material->mMaterialParams.specularTransmission;
		materialConstants.metalRough =
			material->mShadingModel == Material::ShadingModel::MetallicRoughness;
		
		const auto &textures = mMaterialTextures[i];
		// write index -1 if the corresponding texture do not present.
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
	
	commandList->writeBuffer(
		mMaterialConstantsBuffer, mMaterialConstants.data(),
		mMaterialConstants.size() * sizeof(rs::MaterialConstants), 0);
}

void GraphicsScene::writeGeometryBuffer(nvrhi::ICommandList *commandList) {
	/* Fill mesh data buffer on host. */
	/* Normally, a instance is from a mesh, which may contain several geometries.
		In kiraray, we simply ignore this (i.e. the concept of geometry and instances). */
	auto meshes = mScene.lock()->getMeshes();
	for (int i = 0; i < meshes.size(); i++) {
		const auto &mesh = meshes[i];
		rs::MeshData meshData;
		meshData.materialIndex = mesh->getMaterial()->getMaterialId();
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
	commandList->writeBuffer(mMeshDataBuffer, mMeshData.data(),
							 sizeof(rs::MeshData) * meshes.size(), 0);
}

void GraphicsScene::writeInstanceBuffer(nvrhi::ICommandList *commandList) {
	auto instances = mScene.lock()->getMeshInstances();
	for (auto instance : instances) {
		rs::InstanceData instanceData;
		instanceData.transform = instance->getNode()->getGlobalTransform().matrix();
		instanceData.meshIndex = instance->getMesh()->getMeshId();
		mInstanceData.push_back(instanceData);
	}
	commandList->writeBuffer(mInstanceDataBuffer, mInstanceData.data(), sizeof(rs::InstanceData) * instances.size(), 0);
}

void GraphicsScene::writeLightBuffer(nvrhi::ICommandList *commandList) {
	const auto &lights = mScene.lock()->getLights();
	mLightData.clear();
	for (auto light : lights) {
		rs::LightData lightData;
		lightData.type		= light->getType();
		lightData.position	= light->getPosition();
		lightData.direction = light->getDirection();
		lightData.scale		= light->getScale();
		lightData.color		= light->getColor();
		lightData.texture	= 0;
		mLightData.push_back(lightData);
	}
	commandList->writeBuffer(mLightDataBuffer, mLightData.data(),
							 sizeof(rs::LightData) * lights.size());
}

void GraphicsScene::update() {
	bool graphChanged{false}, materialsChanged{false}, lightsChanged{false};
	auto lastUpdates = mScene.lock()->getSceneGraph()->getLastUpdateRecord();
	if ((lastUpdates.updateFlags & SceneGraphNode::UpdateFlags::SubgraphTransform) !=
			SceneGraphNode::UpdateFlags::None &&
		mLastUpdatedFrame < lastUpdates.frameIndex) {
		// update instance transformations
		auto instances = mScene.lock()->getMeshInstances();
		for (int i = 0; i < instances.size(); i++) {
			rs::InstanceData &instanceData = mInstanceData[i];
			instanceData.transform		   = instances[i]->getNode()->getGlobalTransform().matrix();
			instanceData.meshIndex		   = instances[i]->getMesh()->getMeshId();
		}

		graphChanged	 = true;
		mLastUpdatedFrame = lastUpdates.frameIndex;
	}

	const auto &materials = mScene.lock()->getMaterials();
	for (int materialId = 0; materialId < materials.size(); materialId++) {
		const auto &material = materials[materialId];
		if (material->isUpdated()) {
			materialsChanged						 = true; 
			rs::MaterialConstants &materialConstants = mMaterialConstants[materialId];
			materialConstants.baseColor				 = material->mMaterialParams.diffuse;
			materialConstants.specularColor			 = material->mMaterialParams.specular;
			materialConstants.IoR					 = material->mMaterialParams.IoR;
			materialConstants.opacity = material->mMaterialParams.specularTransmission;
			materialConstants.metalRough =
				material->mShadingModel == Material::ShadingModel::MetallicRoughness;
			material->setUpdated(false);
		}
	}
	
	const auto &lights = mScene.lock()->getLights();
	for (int lightId = 0; lightId < lights.size(); lightId++) {
		const auto &light = lights[lightId];
		if (light->isUpdated()) {
			lightsChanged			 = true;
			rs::LightData &lightData = mLightData[lightId];
			lightData.position		 = light->getPosition();
			lightData.direction		 = light->getDirection();
			lightData.scale			 = light->getScale();
			lightData.color			 = light->getColor();
			light->setUpdated(false);
		}
	}
	
	if (graphChanged || materialsChanged || lightsChanged) {
		mCommandList->open();
		/* write changed buffers... */
		if (graphChanged)
			mCommandList->writeBuffer(mInstanceDataBuffer, mInstanceData.data(),
				sizeof(rs::InstanceData) * mScene.lock()->getMeshInstances().size());
		if (materialsChanged)
			mCommandList->writeBuffer(mMaterialConstantsBuffer, mMaterialConstants.data(),
									  mMaterialConstants.size() * sizeof(rs::MaterialConstants));
		if (lightsChanged)
			mCommandList->writeBuffer(mLightDataBuffer, mLightData.data(),
									  sizeof(rs::LightData) * lights.size());
		mCommandList->close();
		mDevice->executeCommandList(mCommandList);
	}
}

NAMESPACE_END(krr)
