#include "gbuffer.h"
#include "window.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

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
	PROFILE("GBuffer drawing");
	static LaunchParameters <GBufferPass> launchParams = {};
	launchParams.frameIndex					= getFrameIndex();
	launchParams.frameSize					= getFrameSize();
	launchParams.cameraData					= mScene->getCamera()->getCameraData();
	launchParams.sceneData					= mScene->getSceneRT()->getSceneData();
	launchParams.traversable				= mOptixBackend->getRootTraversable();

	mOptixBackend->launch(launchParams, "Primary", getFrameSize()[0], getFrameSize()[1], 
		1, KRR_DEFAULT_STREAM);
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

NAMESPACE_END(krr)