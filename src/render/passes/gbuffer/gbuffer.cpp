#include "gbuffer.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

extern "C" char GBUFFER_PTX[];

void GBufferPass::initialize() {
	if (!mOptixBackend) {
		mOptixBackend = std::make_shared<OptixBackend>();
		mOptixBackend->initialize(OptixInitializeParameters()
									  .setPTX(GBUFFER_PTX)
									  .addRayType("Primary", true, false, false)
									  .addRaygenEntry("Primary"));
	}
}

void GBufferPass::setScene(Scene::SharedPtr scene) { 
	initialize();
	mScene = scene; 

}

void GBufferPass::render(RenderContext* context) {

}

void GBufferPass::renderUI() {
	ui::BulletText("Enabled components");
	ui::Checkbox("Depth", &mEnableDepth);
	ui::Checkbox("Diffuse", &mEnableDiffuse);
	ui::Checkbox("Specular", &mEnableSpecular);
	ui::Checkbox("Normal", &mEnableNormal);
	ui::Checkbox("Emissive", &mEnableEmissive);
	ui::Checkbox("Motion", &mEnableMotion);
}

KRR_NAMESPACE_END