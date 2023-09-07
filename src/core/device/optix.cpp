#include <regex>
#include <optix_function_table_definition.h>

#include "logger.h"
#include "context.h"
#include "optix.h"
#include "util/check.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

OptixPipelineCompileOptions OptixBackend::getPipelineCompileOptions() {
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	// [TODO] check this: currently we want single-level instancing only.
	pipelineCompileOptions.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.usesMotionBlur			   = false;
	pipelineCompileOptions.numPayloadValues			   = 3; /* This restricts maximum number of payload to 3. */
	pipelineCompileOptions.numAttributeValues		   = 0;
	pipelineCompileOptions.exceptionFlags =
		OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	return pipelineCompileOptions;
}

OptixModule OptixBackend::createOptixModule(OptixDeviceContext optixContext,
													 const char *ptx) {
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef KRR_DEBUG_BUILD
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#if (OPTIX_VERSION >= 70400)
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
#else
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
	OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();

#if (OPTIX_VERSION >= 70700)
#define OPTIX_MODULE_CREATE optixModuleCreate
#else
#define OPTIX_MODULE_CREATE optixModuleCreateFromPTX
#endif

	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	OptixModule optixModule;
	OPTIX_CHECK_WITH_LOG(OPTIX_MODULE_CREATE(
		optixContext, &moduleCompileOptions, &pipelineCompileOptions,
		ptx, strlen(ptx), log, &logSize, &optixModule),
		log);
	logDebug(log);
	return optixModule;
}

OptixProgramGroup OptixBackend::createRaygenPG(OptixDeviceContext optixContext,
														OptixModule optixModule,
														const char *entrypoint) {
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	desc.raygen.module = optixModule;
	desc.raygen.entryFunctionName = entrypoint;

	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	OptixProgramGroup pg;
	OPTIX_CHECK_WITH_LOG(
		optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
		log);
	logDebug(log);
	return pg;
}

OptixProgramGroup OptixBackend::createMissPG(OptixDeviceContext optixContext,
													  OptixModule optixModule,
													  const char *entrypoint) {
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	desc.miss.module = optixModule;
	desc.miss.entryFunctionName = entrypoint;

	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	OptixProgramGroup pg;
	OPTIX_CHECK_WITH_LOG(
		optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
		log);
	logDebug(log);

	return pg;
}

OptixProgramGroup OptixBackend::createIntersectionPG(OptixDeviceContext optixContext,
															  OptixModule optixModule,
															  const char *closest, const char *any,
															  const char *intersect) {
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc desc = {};
	desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	if (closest) {
		desc.hitgroup.moduleCH = optixModule;
		desc.hitgroup.entryFunctionNameCH = closest;
	}
	if (any) {
		desc.hitgroup.moduleAH = optixModule;
		desc.hitgroup.entryFunctionNameAH = any;
	}
	if (intersect) {
		desc.hitgroup.moduleIS = optixModule;
		desc.hitgroup.entryFunctionNameIS = intersect;
	}

	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	OptixProgramGroup pg;
	OPTIX_CHECK_WITH_LOG(
		optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
		log);
	logDebug(log);

	return pg;
}

OptixTraversableHandle
OptixScene::buildASFromInputs(OptixDeviceContext optixContext, CUstream cudaStream,
										 const std::vector<OptixBuildInput> &buildInputs,
										 CUDABuffer &accelBuffer, bool compact, bool update) {
	if (buildInputs.empty()) return {};
	if (compact && update)	// updating compacted AS is a to-do.
		Log(Error, "Currently OptixBackend do not support update a compacted AS.");

	// Figure out memory requirements.
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags				= OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	if (compact) accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION; 
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, buildInputs.data(),
											 buildInputs.size(), &blasBufferSizes));

	void *tempBuffer;
	size_t tempSizeInBytes =
		update ? blasBufferSizes.tempUpdateSizeInBytes : blasBufferSizes.tempSizeInBytes;
	CUDA_CHECK(cudaMalloc(&tempBuffer, tempSizeInBytes));
	
	OptixTraversableHandle traversableHandle{0};

	if (compact) {
		uint64_t *compactedSizePtr;
		CUDA_CHECK(cudaMalloc(&compactedSizePtr, sizeof(uint64_t)));
		OptixAccelEmitDesc emitDesc;
		emitDesc.type	= OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = (CUdeviceptr) compactedSizePtr;

		void *uncompactedBuffer;
		CUDA_CHECK(cudaMalloc(&uncompactedBuffer, blasBufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			optixContext, cudaStream, &accelOptions, buildInputs.data(), buildInputs.size(),
			CUdeviceptr(tempBuffer), tempSizeInBytes, CUdeviceptr(uncompactedBuffer),
			blasBufferSizes.outputSizeInBytes, &traversableHandle, &emitDesc, 1));

		CUDA_CHECK(cudaStreamSynchronize(cudaStream));

		uint64_t compactedSize;
		CUDA_CHECK(cudaMemcpyAsync(&compactedSize, compactedSizePtr, sizeof(uint64_t),
								   cudaMemcpyDeviceToHost, cudaStream));
		CUDA_CHECK(cudaFree(compactedSizePtr));
		CUDA_CHECK(cudaStreamSynchronize(cudaStream));

		// Compact the acceleration structure
		accelBuffer.resize(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext, cudaStream, traversableHandle,
									  CUdeviceptr(accelBuffer.data()), compactedSize,
									  &traversableHandle));
		CUDA_CHECK(cudaFree(uncompactedBuffer));
		
	} else {
		accelBuffer.resize(blasBufferSizes.outputSizeInBytes);
		OPTIX_CHECK(optixAccelBuild(
			optixContext, cudaStream, &accelOptions, buildInputs.data(), buildInputs.size(),
			CUdeviceptr(tempBuffer), tempSizeInBytes, CUdeviceptr(accelBuffer.data()),
			blasBufferSizes.outputSizeInBytes, &traversableHandle, nullptr, 0));
	}
	CUDA_CHECK(cudaFree(tempBuffer));
	CUDA_CHECK(cudaStreamSynchronize(cudaStream));
	return traversableHandle;
}

OptixTraversableHandle OptixScene::buildTriangleMeshGAS(OptixDeviceContext optixContext,
																   CUstream cudaStream,
																   const rt::MeshData &mesh,
																   CUDABuffer &accelBuffer) {
	// [TODO] Change scene structure to Mesh->Multi-Geometries, each geometry could have its own material
	// This could be done by specifying different SBT offset (HG records) for each geometry (?)
	// i.e. each geometry -> one build input -> one SBT record.
	OptixBuildInput triangleInputs;
	uint triangleInputFlags;
	CUdeviceptr vertexBufferPtr;
	CUdeviceptr indexBufferPtr;
	
	indexBufferPtr  = (CUdeviceptr) mesh.indices.data();
	vertexBufferPtr = (CUdeviceptr) mesh.positions.data();

	triangleInputs		= {};
	triangleInputs.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	// vertex data desc
	triangleInputs.triangleArray.vertexFormat		 = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInputs.triangleArray.vertexStrideInBytes = sizeof(Vector3f);
	triangleInputs.triangleArray.numVertices		 = mesh.positions.size();
	triangleInputs.triangleArray.vertexBuffers		 = &vertexBufferPtr;
	// index data desc
	triangleInputs.triangleArray.indexFormat		= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInputs.triangleArray.indexStrideInBytes = sizeof(Vector3i);
	triangleInputs.triangleArray.numIndexTriplets	= mesh.indices.size();
	triangleInputs.triangleArray.indexBuffer		= indexBufferPtr;

	triangleInputFlags = OPTIX_GEOMETRY_FLAG_NONE;

	triangleInputs.triangleArray.flags						 = &triangleInputFlags;
	triangleInputs.triangleArray.numSbtRecords				 = 1;
	triangleInputs.triangleArray.sbtIndexOffsetBuffer		 = 0;
	triangleInputs.triangleArray.sbtIndexOffsetSizeInBytes	 = 0;
	triangleInputs.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	Log(Debug, "Building GAS for triangle mesh: %zd vertices and %zd faces", 
		mesh.positions.size(), mesh.indices.size());
	return buildASFromInputs(optixContext, cudaStream, {triangleInputs}, accelBuffer, true);
}

OptixProgramGroup OptixBackend::createRaygenPG(const char *entrypoint) const {
	return createRaygenPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptixBackend::createMissPG(const char *entrypoint) const {
	return createMissPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptixBackend::createIntersectionPG(
	const char *closest, const char *any, const char *intersect) const {
	return createIntersectionPG(optixContext, optixModule, closest, any, intersect);
}

void OptixBackend::initialize(const OptixInitializeParameters &params) {
	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	// temporary workaround for setup context
	optixParameters = params;
	optixContext	= gpContext->optixContext;
	cudaStream		= gpContext->cudaStream;
	
	// creating optix module from ptx
	optixModule = createOptixModule(optixContext, params.ptx);
	// creating program groups
	std::vector<OptixProgramGroup> allPGs;
	// creating RAYGEN PG
	for (string raygenEntryName : params.raygenEntries) {
		raygenPGs.push_back(createRaygenPG(
			("__raygen__" + raygenEntryName).c_str()));
		entryPoints.insert({raygenEntryName, entryPoints.size()});
	}
	for (int rayType = 0; rayType < params.rayTypes.size(); rayType++) {
		string rayTypeName				 = params.rayTypes[rayType];
		const auto [hasCH, hasAH, hasIS] = params.rayClosestShaders[rayType];
		// creating MISS PG
		missPGs.push_back(createMissPG(("__miss__" + rayTypeName).c_str()));
		// creating CLOSEST PG
		hitgroupPGs.push_back(createIntersectionPG(
			hasCH ? ("__closesthit__" + rayTypeName).c_str() : nullptr,
			hasAH ? ("__anyhit__" + rayTypeName).c_str() : nullptr,
			hasIS ? ("__intersection__" + rayTypeName).c_str() : nullptr));		
	}
	allPGs.insert(allPGs.end(), raygenPGs.begin(), raygenPGs.end());
	allPGs.insert(allPGs.end(), missPGs.begin(), missPGs.end());
	allPGs.insert(allPGs.end(), hitgroupPGs.begin(), hitgroupPGs.end());
	// creating optix pipeline from all program groups
	OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();
	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth			 = 3;

	OPTIX_CHECK_WITH_LOG(
		optixPipelineCreate(optixContext, &pipelineCompileOptions,
							&pipelineLinkOptions, allPGs.data(), allPGs.size(),
							log, &logSize, &optixPipeline), log);
	OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the
											 stack size for */
										  optixPipeline, 2 * 1024, 2 * 1024, 2 * 1024, 
		2 /* max traversable graph depth */));
	Log(Debug, log);
}

void OptixBackend::setScene(Scene::SharedPtr _scene){
	scene = _scene;
	scene->initializeSceneRT();		// upload bindless scene data to device buffers.
	buildShaderBindingTable();		// SBT[Instances [RayTypes ...] ...]
}

// [TODO] This routine currently do not support rebuild or dynamic update,
// check back later.
void OptixScene::buildAccelStructure() {
	// this is the first time we met...
	const auto &graph			= scene.lock()->getSceneGraph();
	const auto &instances		= scene.lock()->getMeshInstances();
	const auto &meshes			= scene.lock()->getMeshes();
	const auto &sceneDataDevice = getSceneData();

	// build GAS for each mesh
	traversablesGAS.resize(meshes.size());
	accelBuffersGAS.resize(meshes.size());
	for (int idx = 0; idx < meshes.size(); idx++) {
		const auto &mesh = meshes[idx];
		const auto &meshData = scene.lock()->getSceneRT()->getMeshData()[idx];
		// build GAS for each mesh
		traversablesGAS[idx] =
			buildTriangleMeshGAS(gpContext->optixContext, gpContext->cudaStream, meshData, accelBuffersGAS[idx]);
	}

	// fill optix instance arrays
	instancesIAS.resize(instances.size());
	for (int idx = 0; idx < instances.size(); idx++) {
		const auto &instance		   = instances[idx];
		OptixInstance &instanceData	   = instancesIAS[idx];
		Affine3f transform			   = instance->getNode()->getGlobalTransform();
		instanceData.instanceId		   = idx;
		instanceData.sbtOffset		   = idx * OptixBackend::OPTIX_MAX_RAY_TYPES;
		instanceData.visibilityMask	   = 255;
		instanceData.flags			   = OPTIX_INSTANCE_FLAG_NONE;
		instanceData.traversableHandle = traversablesGAS[instance->getMesh()->getMeshId()];
		// [TODO] Check WHY we should use cudaMemcpy here (instead of memcpy on CPU)? 
		// [TODO] invoke 1 cudaMemcpy here.
		cudaMemcpy(instanceData.transform, transform.data(), sizeof(float) * 12,
				   cudaMemcpyHostToDevice);
	}

	// build IAS
	OptixBuildInput iasBuildInput			 = {};
	iasBuildInput.type						 = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	iasBuildInput.instanceArray.numInstances = instances.size();
	iasBuildInput.instanceArray.instances	 = (CUdeviceptr) instancesIAS.data();
	
	Log(Info, "Building root IAS: %zd instances", instances.size());
	traversableIAS = buildASFromInputs(gpContext->optixContext, gpContext->cudaStream,
									   {iasBuildInput}, accelBufferIAS, false);
	CUDA_CHECK(cudaStreamSynchronize(gpContext->cudaStream));
}

// [TODO] Currently supports updating subgraph transforms only.
void OptixScene::update() {
	static size_t lastUpdatedFrame = 0;
	auto lastUpdates			   = scene.lock()->getSceneGraph()->getLastUpdateRecord();
	if ((lastUpdates.updateFlags & SceneGraphNode::UpdateFlags::SubgraphUpdates)
		!= SceneGraphNode::UpdateFlags::None && lastUpdatedFrame < lastUpdates.frameIndex) {		
		updateAccelStructure();
		lastUpdatedFrame = lastUpdates.frameIndex;
	}
}

OptixScene::OptixScene(Scene::SharedPtr _scene) {
	scene = _scene;
	buildAccelStructure();
}

void OptixScene::updateAccelStructure() {
	PROFILE("Update AS");
	const auto &instances = scene.lock()->getMeshInstances();
	// Currently only supports updating subgraph transformations.
	for (int idx = 0; idx < instances.size(); idx++) {
		const auto &instance		   = instances[idx];
		OptixInstance &instanceData	   = instancesIAS[idx];
		Affine3f transform			   = instance->getNode()->getGlobalTransform();
		cudaMemcpy(instanceData.transform, transform.data(), sizeof(float) * 12,
				   cudaMemcpyHostToDevice);
	}

	// build IAS
	OptixBuildInput iasBuildInput			 = {};
	iasBuildInput.type						 = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	iasBuildInput.instanceArray.numInstances = instances.size();
	iasBuildInput.instanceArray.instances	 = (CUdeviceptr) instancesIAS.data();
	Log(Debug, "Building root IAS: %zd instances", instances.size());

	traversableIAS = buildASFromInputs(gpContext->optixContext, gpContext->cudaStream,
									   {iasBuildInput},
									   accelBufferIAS, false, true);
	CUDA_CHECK(cudaStreamSynchronize(gpContext->cudaStream));
}

void OptixBackend::buildShaderBindingTable() {
	size_t nRayTypes = optixParameters.rayTypes.size();
	if (OPTIX_MAX_RAY_TYPES > 2) Log(Fatal, "Currently supports no more than %zd ray types only,"
		"but there are %zd ray types", OPTIX_MAX_RAY_TYPES, nRayTypes);
	else if (nRayTypes == 0) Log(Fatal, "No ray types has been specified!");

	for (const auto [raygenEntry, index] : entryPoints) {
		RaygenRecord raygenRecord = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[index], &raygenRecord));
		raygenRecords.push_back(raygenRecord);
	}
	for (int type = 0; type < nRayTypes; type++) {
		MissRecord missRecord = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[type], &missRecord));
		missRecords.push_back(missRecord);
	}

	const auto &instances	 = scene->getMeshInstances();
	rt::SceneData sceneData = scene->mSceneRT->getSceneData();
	for (uint instanceId = 0; instanceId < instances.size(); instanceId++) {
		for (uint rayType = 0; rayType < nRayTypes; rayType++) {
			HitgroupRecord hitgroupRecord = {};
			rt::InstanceData *instance	  = &sceneData.instances[instanceId];
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayType], &hitgroupRecord));
			hitgroupRecord.data = {instance};
			hitgroupRecords.push_back(hitgroupRecord);
		}
		for (int rayType = nRayTypes; rayType < OPTIX_MAX_RAY_TYPES; rayType++)
			hitgroupRecords.push_back({});
	}

	for (const auto [raygenEntry, index] : entryPoints) {
		OptixShaderBindingTable sbt		= {};
		sbt.raygenRecord				= (CUdeviceptr) &raygenRecords[index];
		sbt.missRecordBase				= (CUdeviceptr) missRecords.data();
		sbt.missRecordCount				= missRecords.size();
		sbt.missRecordStrideInBytes		= sizeof(MissRecord);
		sbt.hitgroupRecordBase			= (CUdeviceptr) hitgroupRecords.data();
		sbt.hitgroupRecordCount			= hitgroupRecords.size();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		SBT.push_back(sbt);
	}
	CUDA_SYNC_CHECK();
}

KRR_NAMESPACE_END