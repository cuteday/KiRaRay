#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"

#include "gpu/buffer.h"
#include "shaders/path.h"
#include "shaders/postprocess.h"

KRR_NAMESPACE_BEGIN

class Renderer{
public:
    using SharedPtr = std::shared_ptr<Renderer>;

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
    void toDevice();

    bool onKeyEvent(const KeyboardEvent& keyEvent);
    bool onMouseEvent(const MouseEvent& mouseEvent);
    void resize(const vec2i &size);
    void render();
    void renderUI();

    Scene::SharedPtr getScene() { return mpScene; }

    void setScene(Scene::SharedPtr scene) {
        mpScene = scene;
        logInfo("#krr: transfering host scene data to device ...");
        toDevice();
        logInfo("#krr: building AS ...");
        buildAS();
        logInfo("#krr: building SBT ...");
        buildSBT();
        logSuccess("Scene set...");
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

    LaunchParamsPT launchParams;
    CUDABuffer   launchParamsBuffer;

    CUDABuffer colorBuffer;
    CUDABuffer accelBuffer;

// Intrinsic scene data and cuda buffers
    Scene::SharedPtr mpScene;

    CUDABuffer materialBuffer;
// render passes
    AccumulatePass::SharedPtr mpAccumulatePass;
    ToneMappingPass::SharedPtr mpToneMappingPass;
};

class RenderApp : public WindowApp{

public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) {
        mpRenderer = Renderer::SharedPtr(new Renderer());
    }

    void resize(const vec2i& size) override {
        //logSuccess("Resizing window size to " + std::to_string(size.x));
        mpRenderer->resize(size);
        WindowApp::resize(size);
    }

    Renderer::SharedPtr getRenderer() { return mpRenderer; }

    virtual void onMouseEvent(io::MouseEvent& mouseEvent) override {
        if(mpRenderer && mpRenderer->onMouseEvent(mouseEvent)) return;
    }
    
    virtual void onKeyEvent(io::KeyboardEvent &keyEvent) override {
        if (mpRenderer && mpRenderer->onKeyEvent(keyEvent))return;
    }

    void render() override {
        mpRenderer->render();
        //mpAccumulatePass->render(mpRenderer->result());
    }

    void renderUI() override{
        ui::Begin(KRR_PROJECT_NAME);
        ui::Text("Hello, world!");
        ui::Text("Window size: %d %d", fbSize.x, fbSize.y);
        mpRenderer->renderUI();
        ui::End();
    }

    void draw() override {
        //mpAccumulatePass->result().copy_to_device(fbPointer, fbSize.x * fbSize.y);
        mpRenderer->result().copy_to_device(fbPointer, fbSize.x * fbSize.y);
        WindowApp::draw();
    }


private:
    Renderer::SharedPtr mpRenderer;

};

KRR_NAMESPACE_END