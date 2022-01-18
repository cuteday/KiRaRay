#pragma once

#include "kiraray.h"
#include "window.h"

#include "gpu/buffer.h"
#include "shaders/LaunchParams.h"

NAMESPACE_KRR_BEGIN

class RendererApp:WindowApp{
public:
	RendererApp();

	void initOptix();

	void createContext();

	void createModule();

	void createRaygenPrograms();

	void createMissPrograms();

	void createHitgroupPrograms();

	void createPipeline();

	void buildSBT();

    void resize(const vec2i &size) override;

    void render() override;

    void downloadPixels(uint32_t h_pixels[]);

protected:
	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

    OptixDeviceContext optixContext;

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;

    CUDABuffer colorBuffer;
};

NAMESPACE_KRR_END