#include "backend.h"
#include "device/cuda.h"
#include "render/profiler/profiler.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

#define OPTIX_LOG_SIZE 4096
extern "C" char PPG_PTX[];
LaunchParamsPPG *launchParams{};

OptiXPPGBackend::OptiXPPGBackend(Scene &scene) { setScene(scene); }

void OptiXPPGBackend::setScene(Scene &scene) {
	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);

	// use global context and stream for now
	optixContext = gpContext->optixContext;
	cudaStream	 = gpContext->cudaStream;

	// creating optix module from ptx
	optixModule = createOptiXModule(optixContext, PPG_PTX);
	// creating program groups
	OptixProgramGroup raygenClosestPG = createRaygenPG("__raygen__Closest");
	OptixProgramGroup raygenShadowPG  = createRaygenPG("__raygen__Shadow");
	OptixProgramGroup missClosestPG	  = createMissPG("__miss__Closest");
	OptixProgramGroup missShadowPG	  = createMissPG("__miss__Shadow");
	OptixProgramGroup hitClosestPG =
		createIntersectionPG("__closesthit__Closest", "__anyhit__Closest", nullptr);
	OptixProgramGroup hitShadowPG = createIntersectionPG(nullptr, "__anyhit__Shadow", nullptr);

	std::vector<OptixProgramGroup> allPGs = { raygenClosestPG, raygenShadowPG, missClosestPG,
											  missShadowPG,	   hitClosestPG,   hitShadowPG };

	// creating optix pipeline from all program groups
	OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();
	OptixPipelineLinkOptions pipelineLinkOptions	   = {};
	pipelineLinkOptions.maxTraceDepth				   = 5;
#ifdef KRR_DEBUG_BUILD
	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
	OPTIX_CHECK_WITH_LOG(optixPipelineCreate(optixContext, &pipelineCompileOptions,
											 &pipelineLinkOptions, allPGs.data(), allPGs.size(),
											 log, &logSize, &optixPipeline),
						 log);
	logDebug(log);

	OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
										  optixPipeline, 4 * 1024, 4 * 1024, 4 * 1024, 1));
	logDebug(log);

	// creating shader binding table...
	RaygenRecord raygenRecord	  = {};
	MissRecord missRecord		  = {};
	HitgroupRecord hitgroupRecord = {};

	OPTIX_CHECK(optixSbtRecordPackHeader(raygenClosestPG, &raygenRecord));
	raygenClosestRecords.push_back(raygenRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenShadowPG, &raygenRecord));
	raygenShadowRecords.push_back(raygenRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(missClosestPG, &missRecord));
	missClosestRecords.push_back(missRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(missShadowPG, &missRecord));
	missShadowRecords.push_back(missRecord);
	for (MeshData &meshData : *scene.mData.meshes) {
		hitgroupRecord.data = { &meshData };
		OPTIX_CHECK(optixSbtRecordPackHeader(hitClosestPG, &hitgroupRecord));
		hitgroupClosestRecords.push_back(hitgroupRecord);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitShadowPG, &hitgroupRecord));
		hitgroupShadowRecords.push_back(hitgroupRecord);
	}

	closestSBT.raygenRecord				   = (CUdeviceptr) raygenClosestRecords.data();
	closestSBT.missRecordBase			   = (CUdeviceptr) missClosestRecords.data();
	closestSBT.missRecordCount			   = 1;
	closestSBT.missRecordStrideInBytes	   = sizeof(MissRecord);
	closestSBT.hitgroupRecordBase		   = (CUdeviceptr) hitgroupClosestRecords.data();
	closestSBT.hitgroupRecordCount		   = hitgroupClosestRecords.size();
	closestSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

	shadowSBT.raygenRecord				  = (CUdeviceptr) raygenShadowRecords.data();
	shadowSBT.missRecordBase			  = (CUdeviceptr) missShadowRecords.data();
	shadowSBT.missRecordCount			  = 1;
	shadowSBT.missRecordStrideInBytes	  = sizeof(MissRecord);
	shadowSBT.hitgroupRecordBase		  = (CUdeviceptr) hitgroupShadowRecords.data();
	shadowSBT.hitgroupRecordCount		  = hitgroupShadowRecords.size();
	shadowSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

	optixTraversable = buildAccelStructure(optixContext, cudaStream, scene);

	if (!launchParams)
		launchParams = gpContext->alloc->new_object<LaunchParamsPPG>();
	launchParams->sceneData	  = scene.mData;
	launchParams->traversable = optixTraversable;
	sceneData				  = scene.mData;
}

void OptiXPPGBackend::traceClosest(int numRays, RayQueue *currentRayQueue,
								   MissRayQueue *missRayQueue, HitLightRayQueue *hitLightRayQueue,
								   ScatterRayQueue *scatterRayQueue, RayQueue *nextRayQueue) {
	if (optixTraversable) {
		PROFILE("Trace intersect rays");
		static LaunchParamsPPG params = {};
		params.traversable			  = optixTraversable;
		params.sceneData			  = sceneData;
		params.currentRayQueue		  = currentRayQueue;
		params.missRayQueue			  = missRayQueue;
		params.hitLightRayQueue		  = hitLightRayQueue;
		params.scatterRayQueue		  = scatterRayQueue;
		params.nextRayQueue			  = nextRayQueue;
		cudaMemcpy(launchParams, &params, sizeof(LaunchParamsPPG), cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr) launchParams,
								sizeof(LaunchParamsPPG), &closestSBT, numRays, 1, 1));
#ifdef KRR_DEBUG_BUILD
		CUDA_SYNC_CHECK();
#endif
	}
}

void OptiXPPGBackend::traceShadow(int numRays, ShadowRayQueue *shadowRayQueue,
								  PixelStateBuffer *pixelState, GuidedPathStateBuffer *guidedState,
								  bool enableTraining) {
	if (optixTraversable) {
		PROFILE("Trace shadow rays");
		static LaunchParamsPPG params = {};
		params.traversable			  = optixTraversable;
		params.sceneData			  = sceneData;
		params.shadowRayQueue		  = shadowRayQueue;
		params.pixelState			  = pixelState;
		params.guidedState			  = guidedState;
		params.enableTraining		  = enableTraining;
		cudaMemcpy(launchParams, &params, sizeof(LaunchParamsPPG), cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr) launchParams,
								sizeof(LaunchParamsPPG), &shadowSBT, numRays, 1, 1));
#ifdef KRR_DEBUG_BUILD
		CUDA_SYNC_CHECK();
#endif
	}
}
KRR_NAMESPACE_END