#include "common.h"

#include "scene.h"
#include "wavefront.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

class OptiXBackend {
public:
	OptiXBackend() = default;

	static OptixModule createOptiXModule(OptixDeviceContext optixContext, const char* ptx);
	static OptixPipelineCompileOptions getPipelineCompileOptions();

	static OptixProgramGroup createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule,
		const char* closest, const char* any, const char* intersect);

	static OptixTraversableHandle buildAccelStructure(OptixDeviceContext optixContext, CUstream cudaStream, Scene& scene);
};

class OptiXWavefrontBackend : public OptiXBackend {
public:
	OptiXWavefrontBackend(Scene& scene);

	void traceClosest();
	void traceShadow();

protected:
	OptixProgramGroup createRaygenPG(const char* entrypoint) const;
	OptixProgramGroup createMissPG(const char* entrypoint) const;
	OptixProgramGroup createIntersectionPG(const char* closest, const char* any,
		const char* intersect) const;

private:
	OptixModule optixModule;
	OptixPipeline optixPipeline;
	OptixDeviceContext optixContext;
	CUstream cudaStream;

	OptixShaderBindingTable closestSBT = {};
	OptixShaderBindingTable shadowSBT = {};
	OptixTraversableHandle optixTraversable = {};

	inter::vector<RaygenRecord> raygenClosestRecords;
	inter::vector<HitgroupRecord> hitgroupClosestRecords;
	inter::vector<MissRecord> missClosestRecords;
	inter::vector<RaygenRecord> raygenShadowRecords;
	inter::vector<HitgroupRecord> hitgroupShadowRecords;
	inter::vector<MissRecord> missShadowRecords;
};

KRR_NAMESPACE_END