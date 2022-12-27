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
	KRR_REGISTER_PASS_DEC(MegakernelPathTracer);

	MegakernelPathTracer() = default;
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

	void initialize();

	void setScene(Scene::SharedPtr scene) override;

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

	friend void to_json(json &j, const MegakernelPathTracer &p) {
		j = json{
			{ "nee", p.launchParams.NEE },
			{ "max_depth", p.launchParams.maxDepth },
			{ "rr", p.launchParams.probRR },
		};
	}

	friend void from_json(const json &j, MegakernelPathTracer &p) {
		j.at("nee").get_to(p.launchParams.NEE);
		j.at("max_depth").get_to(p.launchParams.maxDepth);
		j.at("rr").get_to(p.launchParams.probRR);
	}
};

KRR_NAMESPACE_END