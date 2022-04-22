#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

class WavefrontPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<WavefrontPathTracer>;
	
	void resize(const vec2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;

protected:
	void handleHit();
	void handleMiss();
	void evalDirect();

private:

};

KRR_NAMESPACE_END