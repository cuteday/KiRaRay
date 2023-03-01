#include "logger.h"
#include "context.h"
#include "optix.h"

KRR_NAMESPACE_BEGIN

OptixPipelineCompileOptions OptiXBackend::getPipelineCompileOptions() {
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	// currently we do not implement scene graph and instancing, as such this optimizes performance.
	pipelineCompileOptions.traversableGraphFlags	   = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur		= false;
	pipelineCompileOptions.numPayloadValues			   = 3; /* This restricts maximum number of payload to 3. */
	pipelineCompileOptions.numAttributeValues		   = 0;
	pipelineCompileOptions.exceptionFlags = //OPTIX_EXCEPTION_FLAG_NONE;
		(OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
			OPTIX_EXCEPTION_FLAG_DEBUG);
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	return pipelineCompileOptions;
}

OptixModule OptiXBackend::createOptiXModule(OptixDeviceContext optixContext, const char* ptx) {
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

	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);
	OptixModule optixModule;
	OPTIX_CHECK_WITH_LOG(optixModuleCreateFromPTX(
		optixContext, &moduleCompileOptions, &pipelineCompileOptions,
		ptx, strlen(ptx), log, &logSize, &optixModule),
		log);
	logDebug(log);
	return optixModule;
}

OptixProgramGroup OptiXBackend::createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint) {
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

OptixProgramGroup OptiXBackend::createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint) {
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

OptixProgramGroup OptiXBackend::createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* closest, const char* any, const char* intersect) {
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

OptixTraversableHandle OptiXBackend::buildAccelStructure(
	OptixDeviceContext optixContext, CUstream cudaStream, Scene& scene){
	auto &sceneData			  = scene.mpSceneRT->getSceneData();
	std::vector<Mesh>& meshes = scene.meshes;
	std::vector<OptixBuildInput> triangleInputs(meshes.size());
	std::vector<uint> triangleInputFlags(meshes.size());
	std::vector<CUdeviceptr> vertexBufferPtr(meshes.size());
	std::vector<CUdeviceptr> indexBufferPtr(meshes.size());

	for (uint i = 0; i < meshes.size(); i++) {
		rt::MeshData &mesh = (*sceneData.meshes)[i];

		indexBufferPtr[i]  = (CUdeviceptr)mesh.indices.data();
		vertexBufferPtr[i] = (CUdeviceptr)mesh.vertices.data();

		triangleInputs[i] = {};
		triangleInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		// vertex data desc
		triangleInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInputs[i].triangleArray.vertexStrideInBytes = sizeof(VertexAttribute);
		triangleInputs[i].triangleArray.numVertices = mesh.vertices.size();
		triangleInputs[i].triangleArray.vertexBuffers = &vertexBufferPtr[i];
		// index data desc
		triangleInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInputs[i].triangleArray.indexStrideInBytes = sizeof(Vector3i);
		triangleInputs[i].triangleArray.numIndexTriplets = mesh.indices.size();
		triangleInputs[i].triangleArray.indexBuffer = indexBufferPtr[i];
		
		triangleInputFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
		
		triangleInputs[i].triangleArray.flags = &triangleInputFlags[i];
		triangleInputs[i].triangleArray.numSbtRecords = 1;
		triangleInputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE |
		OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes accelBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
		&accelOptions, triangleInputs.data(), meshes.size(), &accelBufferSizes));

	// prepare for compaction
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.resize(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.data();

	// building process
	CUDABuffer tempBuffer;
	tempBuffer.resize(accelBufferSizes.tempSizeInBytes);
	CUDABuffer outputBuffer;
	outputBuffer.resize(accelBufferSizes.outputSizeInBytes);

	OptixTraversableHandle asHandle = {};

	OPTIX_CHECK(optixAccelBuild(optixContext,
		cudaStream,
		&accelOptions,
		triangleInputs.data(),
		meshes.size(),
		tempBuffer.data(),
		tempBuffer.size(),
		outputBuffer.data(),
		outputBuffer.size(),
		&asHandle,
		&emitDesc,
		1));
	CUDA_SYNC_CHECK();

	// perform compaction
	uint64_t compactedSize;
	compactedSizeBuffer.copy_to_host(&compactedSize, 1);

	static CUDABuffer accelBuffer;
	accelBuffer.resize(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext,
		cudaStream,
		asHandle,
		accelBuffer.data(),
		accelBuffer.size(),
		&asHandle));
	CUDA_SYNC_CHECK();
	return asHandle;
}

KRR_NAMESPACE_END