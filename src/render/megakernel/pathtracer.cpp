#include <optix_types.h>

#include "device/cuda.h"
#include "pathtracer.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

extern "C" char PATHTRACER_PTX[];

void MegakernelPathTracer::compilePrograms() {
	auto params = optixBackend->getParameters();
	params.boundValues.clear();
	params.addBoundValue(offsetof(LaunchParameters<MegakernelPathTracer>, spp),
						 sizeof(launchParams.spp), &launchParams.spp, "Samples per pixel");
	optixBackend->initialize(params);
	needsRecompile = false;
}

void MegakernelPathTracer::resize(const Vector2i &size) {
	launchParams.fbSize = getFrameSize();
	needsRecompile		= true;
}

void MegakernelPathTracer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (!optixBackend) optixBackend = std::make_shared<OptixBackend>();
	auto params = OptixInitializeParameters()
					  .setPTX(PATHTRACER_PTX)
					  .addRaygenEntry("Pathtracer")
					  .addRayType("Radiance", true, true, false)
					  .addRayType("ShadowRay", true, true, false)
					  .setMaxTraversableDepth(scene->getMaxGraphDepth());
	optixBackend->setScene(scene);
	optixBackend->initialize(params);
	needsRecompile = true;
}

void MegakernelPathTracer::renderUI() {
	ui::Text("Path tracing parameters");
	ui::SliderFloat("RR absorption probability", &launchParams.probRR, 0.f, 1.f, "%.3f");
	ui::InputInt("Max bounces", &launchParams.maxDepth);
	ui::DragFloat("Radiance clip", &launchParams.clampThreshold, 0.1, 1, 500);
	needsRecompile |= ui::InputInt("Samples per pixel", &launchParams.spp, 1, 1);
	ui::Checkbox("Next event estimation", &launchParams.NEE);
	if (launchParams.NEE) 
		ui::InputInt("Light sample count", &launchParams.lightSamples);
	ui::Text("Debugging");
	ui::Checkbox("Shader debug output", &launchParams.debugOutput);
	ui::InputInt2("Debug pixel", (int *) &launchParams.debugPixel);
}

void MegakernelPathTracer::render(RenderContext *context) {
	if (getFrameSize().isZero()) return;
	launchParams.colorSpace	 = KRR_DEFAULT_COLORSPACE;
	launchParams.colorBuffer = context->getColorTexture()->getCudaRenderTarget();
	launchParams.camera		 = mScene->getCamera()->getCameraData();
	launchParams.sceneData	 = mScene->getSceneRT()->getSceneData();
	launchParams.traversable = optixBackend->getRootTraversable();
	launchParams.frameID	 = (uint) getFrameIndex();
	{
		PROFILE("Megakernel Path Tracer");
		if (needsRecompile) compilePrograms();
		optixBackend->launch(launchParams, "Pathtracer", getFrameSize()[0], getFrameSize()[1],
			1, KRR_DEFAULT_STREAM);
	}
}

KRR_REGISTER_PASS_DEF(MegakernelPathTracer);
NAMESPACE_END(krr)