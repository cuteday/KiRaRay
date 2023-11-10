#include <optix_types.h>

#include "device/cuda.h"
#include "integrator.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

extern "C" char BDPT_PTX[];

namespace {
	LaunchParameters<BDPTIntegrator> *launchParamsDevice{};
}

BDPTIntegrator::BDPTIntegrator() {
	logFatal("Bidirectional path tracer is currently WIP");
}

BDPTIntegrator::~BDPTIntegrator() {}

void BDPTIntegrator::renderUI() {
	ui::Text("Render parameters");
	ui::SliderFloat("RR absorption probability", &launchParams.probRR, 0.f, 1.f, "%.3f");
	ui::InputInt("Max depth", &launchParams.maxDepth);
	ui::DragFloat("Radiance clip", &launchParams.clampThreshold, 0.1, 1, 500);
	
	if (ui::CollapsingHeader("Debugging")) {
		ui::Checkbox("Shader debug output", &launchParams.debugOutput);
		if (launchParams.debugOutput)
			ui::InputInt2("Debug pixel", (int *) &launchParams.debugPixel);
	}
}

void BDPTIntegrator::render(RenderContext *context) {
	if (getFrameSize()[0] * getFrameSize()[1] == 0)
		return;
	PROFILE("BDPT Integrator");
	{
		PROFILE("Path tracing kernel");
	}
	CUDA_SYNC_CHECK();
}

KRR_REGISTER_PASS_DEF(BDPTIntegrator);
KRR_NAMESPACE_END