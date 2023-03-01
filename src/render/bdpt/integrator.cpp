#include <optix_types.h>

#include "device/cuda.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "render/wavefront/backend.h"

KRR_NAMESPACE_BEGIN

extern "C" char BDPT_PTX[];

namespace {
	LaunchParamsBDPT *launchParamsDevice{};
}

BDPTIntegrator::BDPTIntegrator() {
	logFatal("Bidirectional path tracer is currently WIP");
	logDebug("Setting up module ...");
	module = OptiXBackend::createOptiXModule(gpContext->optixContext, BDPT_PTX);
	logDebug("Creating program groups ...");
	createProgramGroups();
	logDebug("Setting up optix pipeline ...");
	createPipeline();
	if (launchParamsDevice == nullptr)
		launchParamsDevice = gpContext->alloc->new_object<LaunchParamsBDPT>();
	logSuccess("BDPT Integrator::Optix context fully set up");
}

BDPTIntegrator::~BDPTIntegrator() {
	if (launchParamsDevice)
		gpContext->alloc->deallocate_object(launchParamsDevice);
}

void BDPTIntegrator::createProgramGroups() {
	// setup raygen program groups
}

void BDPTIntegrator::createPipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	OptixPipelineCompileOptions pipelineCompileOptions = OptiXBackend::getPipelineCompileOptions();
	OptixPipelineLinkOptions pipelineLinkOptions	   = {};
	pipelineLinkOptions.maxTraceDepth				   = 3;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(gpContext->optixContext, &pipelineCompileOptions,
									&pipelineLinkOptions, programGroups.data(),
									(int) programGroups.size(), log, &sizeof_log, &pipeline));
	logDebug(log);
}

void BDPTIntegrator::buildSBT() {
	// build raygen records
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		raygenRecords.push_back(rec);
	}
	sbt.raygenRecord = (CUdeviceptr) raygenRecords.data();

	// build miss records
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		missRecords.push_back(rec);
	}
	sbt.missRecordBase			= (CUdeviceptr) missRecords.data();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount			= (int) missRecords.size();

	// build hitgroup records
	uint numMeshes = mpScene->meshes.size();
	rt::SceneData &sceneData = mpScene->mpSceneRT->getSceneData();
	
	for (uint meshId = 0; meshId < numMeshes; meshId++) {
		for (uint rayId = 0; rayId < 3; rayId++) {
			HitgroupRecord rec;
			rt::MeshData *mesh = &(*sceneData.meshes)[meshId];
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayId], &rec));
			rec.data = { mesh };
			hitgroupRecords.push_back(rec);
		}
	}
	sbt.hitgroupRecordBase			= (CUdeviceptr) hitgroupRecords.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount			= hitgroupRecords.size();
}

void BDPTIntegrator::buildAS() {
	OptixTraversableHandle &asHandle = launchParams.traversable;
	asHandle =
		OptiXBackend::buildAccelStructure(gpContext->optixContext, gpContext->cudaStream, *mpScene);
}

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

void BDPTIntegrator::render(RenderFrame::SharedPtr frame) {
	if (mFrameSize[0] * mFrameSize[1] == 0)
		return;
	PROFILE("BDPT Integrator");
	{
		PROFILE("Updating parameters");
		CUDATrackedMemory::singleton.PrefetchToGPU();
		launchParams.fbSize		 = mFrameSize;
		launchParams.colorBuffer = frame->getCudaRenderTarget();
		launchParams.camera		 = mpScene->getCamera();
		launchParams.sceneData	 = mpScene->mpSceneRT->getSceneData();
		launchParams.frameID++;
		cudaMemcpy(launchParamsDevice, &launchParams, sizeof(LaunchParamsBDPT),
				   cudaMemcpyHostToDevice);
	}
	{
		PROFILE("Path tracing kernel");
		OPTIX_CHECK(optixLaunch(pipeline, gpContext->cudaStream, (CUdeviceptr) launchParamsDevice,
								sizeof(LaunchParamsBDPT), &sbt, launchParams.fbSize[0],
								launchParams.fbSize[1], 1));
	}
	CUDA_SYNC_CHECK();
}

KRR_REGISTER_PASS_DEF(BDPTIntegrator);
KRR_NAMESPACE_END