#pragma once
#include "common.h"

#include "scene.h"
#include "workqueue.h"
#include "wavefront.h"

#include "device/optix.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN


class OptiXWavefrontBackend : public OptiXBackend {
public:
	OptiXWavefrontBackend() = default;
	OptiXWavefrontBackend(Scene& scene);
	void setScene(Scene& scene);

	void traceClosest(int numRays,
		RayQueue* currentRayQueue,
		MissRayQueue* missRayQueue,
		HitLightRayQueue* hitLightRayQueue,
		ScatterRayQueue* scatterRayQueue,
		RayQueue* nextRayQueue);

	void traceShadow(int numRays,
		ShadowRayQueue* shadowRayQueue,
		PixelStateBuffer* pixelState);

protected:
	OptixProgramGroup createRaygenPG(const char *entrypoint) const;
	OptixProgramGroup createMissPG(const char *entrypoint) const;
	OptixProgramGroup createIntersectionPG(const char *closest, const char *any,
										   const char *intersect) const;

	OptixModule optixModule;
	OptixPipeline optixPipeline;
	OptixDeviceContext optixContext;
	CUstream cudaStream;

	OptixShaderBindingTable closestSBT{};
	OptixShaderBindingTable shadowSBT{};
	OptixTraversableHandle optixTraversable{};

	Scene::SceneData sceneData{};

	inter::vector<RaygenRecord> raygenClosestRecords;
	inter::vector<HitgroupRecord> hitgroupClosestRecords;
	inter::vector<MissRecord> missClosestRecords;
	inter::vector<RaygenRecord> raygenShadowRecords;
	inter::vector<HitgroupRecord> hitgroupShadowRecords;
	inter::vector<MissRecord> missShadowRecords;
};

KRR_NAMESPACE_END