#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "device/cuda.h"
#include "pathtracer.h"
#include "render/profiler/profiler.h"
#include "render/wavefront/backend.h"

KRR_NAMESPACE_BEGIN

extern "C" char PATHTRACER_PTX[];

void MegakernelPathTracer::initialize() {
	if (!optixBackend) {
		optixBackend = new OptiXBackendImpl();
		auto params	 = OptiXInitializeParameters()
						  .setPTX(PATHTRACER_PTX)
						  .addRayType("Radiance", true, true, false)
						  .addRayType("ShadowRay", true, true, false)
						  .addRaygenEntry("Pathtracer");
		optixBackend->initialize(params);
	}
}

MegakernelPathTracer::~MegakernelPathTracer() { delete optixBackend; }

void MegakernelPathTracer::setScene(Scene::SharedPtr scene) {
	initialize();
	mpScene = scene;
	mpScene->initializeSceneRT();
	optixBackend->setScene(*scene);
}

void MegakernelPathTracer::renderUI() {
	ui::Text("Path tracing parameters");
	ui::InputInt("Samples per pixel", &launchParams.spp);
	ui::SliderFloat("RR absorption probability", &launchParams.probRR, 0.f, 1.f, "%.3f");
	ui::InputInt("Max bounces", &launchParams.maxDepth);
	ui::DragFloat("Radiance clip", &launchParams.clampThreshold, 0.1, 1, 500);
	ui::Checkbox("Next event estimation", &launchParams.NEE);
	if (launchParams.NEE) {
		ui::InputInt("Light sample count", &launchParams.lightSamples);
	}
	ui::Text("Debugging");
	ui::Checkbox("Shader debug output", &launchParams.debugOutput);
	ui::InputInt2("Debug pixel", (int *) &launchParams.debugPixel);
}

void MegakernelPathTracer::render(RenderFrame::SharedPtr frame) {
	if (mFrameSize[0] * mFrameSize[1] == 0)
		return;
	PROFILE("Megakernel Path Tracer");
	{
		PROFILE("Updating parameters");
		launchParams.fbSize		 = mFrameSize;
		launchParams.colorBuffer = frame->getCudaRenderTarget();
		launchParams.camera		 = mpScene->getCamera();
		launchParams.sceneData	 = mpScene->mpSceneRT->getSceneData();
		launchParams.traversable = optixBackend->getRootTraversable();
		launchParams.frameID++;
	}
	{
		PROFILE("Path tracing kernel");
		optixBackend->launch(launchParams, "Pathtracer", mFrameSize[0], mFrameSize[1]);
	}
	CUDA_SYNC_CHECK();
}

KRR_REGISTER_PASS_DEF(MegakernelPathTracer);
KRR_NAMESPACE_END