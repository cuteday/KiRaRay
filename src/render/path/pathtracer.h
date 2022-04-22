#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/buffer.h"
#include "device/context.h"
#include "render/path/path.h"

KRR_NAMESPACE_BEGIN

class PathTracer: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<PathTracer>;

	PathTracer();

	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();
	void buildAS();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override;
	bool onMouseEvent(const MouseEvent& mouseEvent) override;
	void renderUI() override;
	void render(CUDABuffer& frame) override;
	void resize(const vec2i& size) override { 
		mFrameSize = size;
		launchParams.fbSize = size; 
	}

	void setScene(Scene::SharedPtr scene) override {
		mpScene = scene;
		mpScene->toDevice();
		buildAS();
		buildSBT();
		logSuccess("Scene set...");
	}

private:
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
	CUDABuffer accelBuffer;
};

KRR_NAMESPACE_END