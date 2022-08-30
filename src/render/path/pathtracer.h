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

class MegakernelPathTracer: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<MegakernelPathTracer>;

	MegakernelPathTracer();
	~MegakernelPathTracer();

	void createProgramGroups();
	void createPipeline();
	void buildSBT();
	void buildAS();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(CUDABuffer& frame) override;
	void resize(const Vector2i& size) override { 
		mFrameSize = launchParams.fbSize = size; 
	}

	void setScene(Scene::SharedPtr scene) override {
		mpScene = scene;
		mpScene->toDevice();
		buildAS();
		buildSBT();
		logSuccess("Scene set...");
	}

	string getName() const override { return "MegakernelPathTracer"; }

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
	LaunchParamsPT* launchParamsDevice;
};

KRR_NAMESPACE_END