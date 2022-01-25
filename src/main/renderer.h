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

        logInfo("#krr: building AS ...");
        buildAS();

        logInfo("#krr: building SBT ...");
        buildSBT();

        logSuccess("Scene reset.");
    }

    CUDABuffer& result();

private:

// OptiX and CUDA context
	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

    OptixDeviceContext mOptixContext;

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
    CUDABuffer accelBuffer;

// Intrinsic scene data and cuda buffers
    Scene::SharedPtr mpScene;

    std::vector<CUDABuffer> indexBuffers;
    std::vector<CUDABuffer> vertexBuffers;
};

class RenderApp : public WindowApp{

public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) {}

    void resize(const vec2i& size) override {
        logSuccess("Resizing window size to " + std::to_string(size.x));
        mRenderer.resize(size);
        WindowApp::resize(size);
    }

    Renderer& renderer() { return mRenderer; }

    void render() override {
        mRenderer.render();
    }

    void draw_ui() override{
        ImGui::Begin(KRR_PROJECT_NAME);
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