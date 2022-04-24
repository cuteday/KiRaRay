#include "backend.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

OptixPipelineCompileOptions OptiXBackend::getPipelineCompileOptions() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 4;
    // OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.exceptionFlags = //OPTIX_EXCEPTION_FLAG_NONE;
        (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            OPTIX_EXCEPTION_FLAG_DEBUG);
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
    return pipelineCompileOptions;
}

OptixModule OptiXBackend::createOptiXModule(OptixDeviceContext optixContext, const char* ptx) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    // TODO: REVIEW THIS
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#if (OPTIX_VERSION >= 70400)
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif // OPTIX_VERSION
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();

    char log[4096];
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

    char log[4096];
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

    char log[4096];
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

    char log[4096];
    size_t logSize = sizeof(log);
    OptixProgramGroup pg;
    OPTIX_CHECK_WITH_LOG(
        optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
        log);
    logDebug(log);

    return pg;
}

OptixTraversableHandle OptiXBackend::buildAccelStructure(Scene& scene, CUDABuffer& asBuffer)
{
    std::vector<Mesh>& meshes = scene.meshes;
    std::vector<OptixBuildInput> triangleInputs(meshes.size());
    std::vector<uint> triangleInputFlags(meshes.size());
    std::vector<CUdeviceptr> vertexBufferPtr(meshes.size());
    std::vector<CUdeviceptr> indexBufferPtr(meshes.size());

    for (uint i = 0; i < meshes.size(); i++) {
        Mesh& mesh = meshes[i];

        indexBufferPtr[i] = (CUdeviceptr)mesh.indices.data();
        vertexBufferPtr[i] = (CUdeviceptr)mesh.vertices.data();

        triangleInputs[i] = {};
        triangleInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        // vertex data desc
        triangleInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInputs[i].triangleArray.vertexStrideInBytes = sizeof(vec3f);
        triangleInputs[i].triangleArray.numVertices = mesh.vertices.size();
        triangleInputs[i].triangleArray.vertexBuffers = &vertexBufferPtr[i];
        // index data desc
        triangleInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInputs[i].triangleArray.indexStrideInBytes = sizeof(vec3i);
        triangleInputs[i].triangleArray.numIndexTriplets = mesh.indices.size();
        triangleInputs[i].triangleArray.indexBuffer = indexBufferPtr[i];
        triangleInputFlags[i] = 0;
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

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(gpContext->optixContext,
        &accelOptions, triangleInputs.data(), meshes.size(), &blasBufferSizes));

    // prepare for compaction
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.resize(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.data();

    // building process
    CUDABuffer tempBuffer;
    tempBuffer.resize(blasBufferSizes.tempSizeInBytes);
    CUDABuffer outputBuffer;
    outputBuffer.resize(blasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle asHandle = {};

    OPTIX_CHECK(optixAccelBuild(gpContext->optixContext,
        gpContext->cudaStream,
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

    asBuffer.resize(compactedSize);
    OPTIX_CHECK(optixAccelCompact(gpContext->optixContext,
        gpContext->cudaStream,
        asHandle,
        asBuffer.data(),
        //CUdeviceptr(asBuffer),
        asBuffer.size(),
        &asHandle));
    CUDA_SYNC_CHECK();
    return asHandle;
}

OptixProgramGroup OptiXWavefrontBackend::createRaygenPG(const char* entrypoint) const{
    return OptiXBackend::createRaygenPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptiXWavefrontBackend::createMissPG(const char* entrypoint) const{
    return OptiXBackend::createMissPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptiXWavefrontBackend::createIntersectionPG(const char* closest, const char* any, const char* intersect) const{
    return OptiXBackend::createIntersectionPG(optixContext, optixModule, closest, any, intersect);
}

KRR_NAMESPACE_END