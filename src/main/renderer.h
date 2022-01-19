#pragma once

#include "kiraray.h"
#include "window.h"

#include "gpu/buffer.h"
#include "shaders/LaunchParams.h"

NAMESPACE_KRR_BEGIN

class Renderer{
public:
    Renderer();

    void initOptix();

	void createContext();

	void createModule();

	void createRaygenPrograms();

	void createMissPrograms();

	void createHitgroupPrograms();

	void createPipeline();

	void buildSBT();

    void resize(const vec2i &size);

    void render();

    CUDABuffer& result();

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

class RenderApp : public WindowApp{

public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) {}

    void resize(const vec2i& size) override {
        renderer.resize(size);
        WindowApp::resize(size);
    }

    void render() override {
        renderer.render();
    }

    void draw() override {
        renderer.result().copy_to_device(fbPointer, fbSize.x * fbSize.y);
        WindowApp::draw();
    }


private:
    Renderer renderer;
};

NAMESPACE_KRR_END