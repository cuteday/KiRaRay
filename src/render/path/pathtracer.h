#pragma once

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/optix.h"
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

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(RenderFrame::SharedPtr frame) override;
	void resize(const Vector2i& size) override { 
		mFrameSize = launchParams.fbSize = size; 
	}

	void initialize();

	void setScene(Scene::SharedPtr scene) override;

	string getName() const override { return "MegakernelPathTracer"; }

private:

	OptiXBackend *optixBackend{nullptr};
	LaunchParamsPT launchParams;

	friend void to_json(json &j, const MegakernelPathTracer &p) {
		j = json{
			{ "nee", p.launchParams.NEE },
			{ "max_depth", p.launchParams.maxDepth },
			{ "rr", p.launchParams.probRR },
		};
	}

	friend void from_json(const json &j, MegakernelPathTracer &p) {
		p.launchParams.NEE = j.value("nee", true);
		p.launchParams.maxDepth = j.value("max_depth", 10);
		p.launchParams.probRR = j.value("rr", 0.8);
	}
};

KRR_NAMESPACE_END