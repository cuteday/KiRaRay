#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "device/cuda.h"
#include "render/wavefront/backend.h"
#include "pathtracer.h"

KRR_NAMESPACE_BEGIN

extern "C" char PATHTRACER_PTX[];

PathTracer::PathTracer()
{
	logInfo("setting up module ...");
	module = OptiXBackend::createOptiXModule(gpContext->optixContext, PATHTRACER_PTX);
	logInfo("creating program groups ...");
	createProgramGroups();
	logInfo("setting up optix pipeline ...");
	createPipeline();
	
	launchParamsBuffer.resize(sizeof(launchParams));
	logSuccess("PathTracer::Optix 7 context fully set up");
}

void PathTracer::createProgramGroups()
{
	// setup raygen program groups
	raygenPGs.resize(1);;
	raygenPGs[0] = OptiXBackend::createRaygenPG(gpContext->optixContext, module, "__raygen__Pathtracer");
	// setup hit program groups
	missPGs.resize(RAY_TYPE_COUNT);	
	for (int i = 0; i < RAY_TYPE_COUNT; i++) {
		string msFuncName = "__miss__" + shaderProgramNames[i];
		missPGs[i] = OptiXBackend::createMissPG(gpContext->optixContext, module, msFuncName.c_str());
	}
	// setup miss program groups
	hitgroupPGs.resize(RAY_TYPE_COUNT);
	for (int i = 0; i < RAY_TYPE_COUNT; i++) {
		string chFuncName = "__closesthit__" + shaderProgramNames[i];
		hitgroupPGs[i] = OptiXBackend::createIntersectionPG(gpContext->optixContext,
			module, chFuncName.c_str(), nullptr, nullptr);
	}
}

void PathTracer::createPipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	OptixPipelineCompileOptions pipelineCompileOptions = OptiXBackend::getPipelineCompileOptions();
	OptixPipelineLinkOptions pipelineLinkOptions = {}; 
	pipelineLinkOptions.maxTraceDepth = 3;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(gpContext->optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programGroups.data(),
		(int)programGroups.size(),
		log, &sizeof_log,
		&pipeline
	));
	if (sizeof_log > 1) PRINT(log);

	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		pipeline,
		2 * 1024,
		2 * 1024,
		2 * 1024,
		1));
	if (sizeof_log > 1) PRINT(log);
}


void PathTracer::buildSBT()
{
	// build raygen records
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		raygenRecords.push_back(rec);
	}
	sbt.raygenRecord = (CUdeviceptr)raygenRecords.data();

	// build miss records
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		missRecords.push_back(rec);
	}
	sbt.missRecordBase = (CUdeviceptr)missRecords.data();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	// build hitgroup records
	uint numMeshes = mpScene->meshes.size();
	for (uint meshId = 0; meshId < numMeshes; meshId++) {
		for (uint rayId = 0; rayId < RAY_TYPE_COUNT; rayId++) {
			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayId], &rec));
			rec.data = { meshId };
			hitgroupRecords.push_back(rec);
		}
	}
	sbt.hitgroupRecordBase = (CUdeviceptr)hitgroupRecords.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = hitgroupRecords.size();
}

void PathTracer::buildAS()
{
	OptixTraversableHandle& asHandle = launchParams.traversable;
	asHandle = OptiXBackend::buildAccelStructure(gpContext->optixContext, gpContext->cudaStream, *mpScene);
}

void PathTracer::renderUI() {
	if (ui::CollapsingHeader("Path tracer")) {
		ui::Text("Path tracing parameters");
		ui::InputInt("Samples per pixel", &launchParams.spp);
		ui::SliderFloat("RR absorption probability", &launchParams.probRR, 0.f, 1.f, "%.3f");
		ui::SliderInt("Max recursion depth", &launchParams.maxDepth, 0, 30);
		if (mpScene->mData.lights.size() > 0)	// only when we have light sources...
			ui::Checkbox("Next event estimation", &launchParams.NEE);
		if (launchParams.NEE) {
			ui::Checkbox("Multiple importance sampling", &launchParams.MIS);
			ui::InputInt("Light sample count", &launchParams.lightSamples);
		}
		ui::Text("Debugging");
		ui::Checkbox("Shader debug output", &launchParams.debugOutput);
		ui::InputInt2("Debug pixel", (int*)&launchParams.debugPixel);
	}
}

void PathTracer::render(CUDABuffer& frame)
{
	if (launchParams.fbSize.x * launchParams.fbSize.y == 0) return;

	// prefetch memory resource
	CUDATrackedMemory::singleton.PrefetchToGPU();

	launchParams.fbSize = mFrameSize;
	launchParams.colorBuffer = (vec4f*)frame.data();
	memcpy(&launchParams.camera, mpScene->getCamera().get(), sizeof(Camera));
	launchParams.sceneData = mpScene->getSceneData();
	launchParams.frameID++;
	launchParamsBuffer.copy_from_host(&launchParams, 1);

	OPTIX_CHECK(optixLaunch(
		pipeline, gpContext->cudaStream,
		launchParamsBuffer.data(),
		launchParamsBuffer.size(),
		&sbt,
		launchParams.fbSize.x,
		launchParams.fbSize.y,
		1
	));
	CUDA_SYNC_CHECK();
}

KRR_NAMESPACE_END