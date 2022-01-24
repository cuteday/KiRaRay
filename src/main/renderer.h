#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"

#include "gpu/buffer.h"
#include "shaders/LaunchParams.h"

KRR_NAMESPACE_BEGIN

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

    void buildAS();

    void resize(const vec2i &size);

    void render();

    void setScene(Scene::SharedPtr scene) {
        mpScene = scene;
    }

    CUDABuffer& result();

protected:

// OptiX and CUDA context
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

// Intrinsic data
    Scene::SharedPtr mpScene;
};

class RenderApp : public WindowApp{

public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) {}

    void resize(const vec2i& size) override {
        WindowApp::resize(size);
        mRenderer.resize(size);
    }

    Renderer& renderer() { return mRenderer; }

    void render() override {
        mRenderer.render();
    }

    void draw_ui() override{
        ImGui::Begin("KiRaRay");
        ImGui::Text("Hello, world!");
        ImGui::End();
    }

    void draw() override {
        mRenderer.result().copy_to_device(fbPointer, fbSize.x * fbSize.y);
        WindowApp::draw();
    }


private:
    Renderer mRenderer;
};

KRR_NAMESPACE_END