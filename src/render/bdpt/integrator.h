#pragma once

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/optix.h"
#include "device/buffer.h"
#include "device/context.h"
#include "render/bdpt/bdpt.h"

NAMESPACE_BEGIN(krr)

class BDPTIntegrator: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<BDPTIntegrator>;
	KRR_REGISTER_PASS_DEC(BDPTIntegrator);

	BDPTIntegrator();
	~BDPTIntegrator();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
	bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
	void renderUI() override;
	void render(RenderContext *context) override;
	void resize(const Vector2i& size) override {}
	void setScene(Scene::SharedPtr scene) override { mScene = scene; }

	string getName() const override { return "BDPTIntegrator"; }

private:
	OptixPipeline               pipeline;
	OptixModule                 module;

	LaunchParameters<BDPTIntegrator> launchParams;
};

NAMESPACE_END(krr)