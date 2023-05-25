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

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(RenderFrame::SharedPtr frame) override;
	void resize(const Vector2i& size) override { 
		mFrameSize = launchParams.fbSize = size; 
	}

	void setScene(Scene::SharedPtr scene) override {
		mScene = scene;
	}

	string getName() const override { return "BDPTIntegrator"; }

private:
	OptixPipeline               pipeline;
	OptixModule                 module;

	LaunchParamsBDPT launchParams;
};

KRR_NAMESPACE_END