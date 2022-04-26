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

	void createProgramGroups();
	void createPipeline();
	void buildSBT();
	void buildAS();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(CUDABuffer& frame) override;
	void resize(const vec2i& size) override { 
		mFrameSize = launchParams.fbSize = size; 
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
	OptixModule                 module;

	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs;
	inter::vector<MissRecord> missRecords;
	inter::vector<RaygenRecord> raygenRecords;
	inter::vector<HitgroupRecord> hitgroupRecords;
	OptixShaderBindingTable sbt = {};

	LaunchParamsPT launchParams;
	CUDABuffer   launchParamsBuffer;
};

KRR_NAMESPACE_END