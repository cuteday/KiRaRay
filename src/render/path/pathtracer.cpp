#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "device/cuda.h"
#include "render/profiler/profiler.h"
#include "render/wavefront/backend.h"
#include "pathtracer.h"

KRR_NAMESPACE_BEGIN

extern "C" char PATHTRACER_PTX[];

MegakernelPathTracer::MegakernelPathTracer() {
	logDebug("Setting up module ...");
	module = OptiXBackend::createOptiXModule(gpContext->optixContext, PATHTRACER_PTX);
	logDebug("Creating program groups ...");
	createProgramGroups();
	logDebug("Setting up optix pipeline ...");
	createPipeline();
	
	launchParamsDevice = gpContext->alloc->new_object<LaunchParamsPT>();
	logSuccess("Megakernel PathTracer::Optix context fully set up");
}

MegakernelPathTracer::~MegakernelPathTracer() {
	gpContext->alloc->deallocate_object(launchParamsDevice);
}

void MegakernelPathTracer::createProgramGroups() {
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
		string ahFuncName = "__anyhit__" + shaderProgramNames[i];
		hitgroupPGs[i]	  = OptiXBackend::createIntersectionPG(
			   gpContext->optixContext, module, chFuncName.c_str(), ahFuncName.c_str(), nullptr);
	}
}

void MegakernelPathTracer::createPipeline() {
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
	logDebug(log);

	//OPTIX_CHECK(optixPipelineSetStackSize (/* [in] The pipeline to configure the stack size for */
	//	pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));
	//logDebug(log);
}


void MegakernelPathTracer::buildSBT() {
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
			MeshData* mesh = &(*mpScene->mData.meshes)[meshId];
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayId], &rec));
			rec.data = { mesh };
			hitgroupRecords.push_back(rec);
		}
	}
	sbt.hitgroupRecordBase = (CUdeviceptr)hitgroupRecords.data();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = hitgroupRecords.size();
}

void MegakernelPathTracer::buildAS() {
	OptixTraversableHandle& asHandle = launchParams.traversable;
	asHandle = OptiXBackend::buildAccelStructure(gpContext->optixContext, gpContext->cudaStream, *mpScene);
}

void MegakernelPathTracer::renderUI() {
	ui::Text("Path tracing parameters");
	ui::InputInt("Samples per pixel", &launchParams.spp);
	ui::SliderFloat("RR absorption probability", &launchParams.probRR, 0.f, 1.f, "%.3f");
	ui::InputInt("Max bounces", &launchParams.maxDepth);
	ui::DragFloat("Radiance clip", &launchParams.clampThreshold, 0.1, 1, 500);
	ui::Checkbox("Next event estimation", &launchParams.NEE);
	if (launchParams.NEE) {
		ui::Checkbox("Multiple importance sampling", &launchParams.MIS);
		ui::InputInt("Light sample count", &launchParams.lightSamples);
	}
	ui::Text("Debugging");
	ui::Checkbox("Shader debug output", &launchParams.debugOutput);
	ui::InputInt2("Debug pixel", (int*)&launchParams.debugPixel);
}

void MegakernelPathTracer::render(CUDABuffer& frame) {
	if (mFrameSize[0] * mFrameSize[1] == 0) return;
	PROFILE("Megakernel Path Tracer");
	{
		PROFILE("Updating parameters");
		CUDATrackedMemory::singleton.PrefetchToGPU();
		launchParams.fbSize = mFrameSize;
		launchParams.colorBuffer = (Color4f*)frame.data();
		launchParams.camera = mpScene->getCamera();
		launchParams.sceneData = mpScene->getSceneData();
		launchParams.frameID++;
		cudaMemcpy(launchParamsDevice, &launchParams, sizeof(LaunchParamsPT), cudaMemcpyHostToDevice);
	}
	{
		PROFILE("Path tracing kernel");
		OPTIX_CHECK(optixLaunch(
			pipeline, gpContext->cudaStream,
			(CUdeviceptr)launchParamsDevice,
			sizeof(LaunchParamsPT),
			&sbt,
			launchParams.fbSize[0],
			launchParams.fbSize[1],
			1
		));
	}
	CUDA_SYNC_CHECK();
}

KRR_REGISTER_PASS_DEF(MegakernelPathTracer);
KRR_NAMESPACE_END