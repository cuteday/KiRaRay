#pragma once

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/optix.h"
#include "device/buffer.h"
#include "device/context.h"
#include "render/bdpt/bdpt.h"

KRR_NAMESPACE_BEGIN

class BDPTIntegrator: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<BDPTIntegrator>;
	KRR_REGISTER_PASS_DEC(BDPTIntegrator);

	BDPTIntegrator();
	~BDPTIntegrator();

	void createProgramGroups();
	void createPipeline();
	void buildSBT();
	void buildAS();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(RenderFrame::SharedPtr frame) override;
	void resize(const Vector2i& size) override { 
		mFrameSize = launchParams.fbSize = size; 
	}

	void setScene(Scene::SharedPtr scene) override {
		mpScene = scene;
		mpScene->initializeSceneRT();
		buildAS();
		buildSBT();
		logSuccess("Scene set...");
	}

	string getName() const override { return "BDPTIntegrator"; }

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
	LaunchParamsBDPT launchParams;
};

KRR_NAMESPACE_END