#include "common.h"

#include "scene.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

class OptiXBackend {
public:
	OptiXBackend() = delete;

	static OptixModule createOptiXModule(OptixDeviceContext optixContext, const char* ptx);
	static OptixPipelineCompileOptions getPipelineCompileOptions();

	static OptixProgramGroup createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule,
		const char* closest, const char* any, const char* intersect);

	static OptixTraversableHandle buildAccelStructure(Scene& scene, CUDABuffer& asBuffer);
};

class OptiXWavefrontBackend : public OptiXBackend {
public:
	OptiXWavefrontBackend(const Scene& scene);

	void IntersectClosest();
	void IntersectShadow();

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

	OptixShaderBindingTable intersectSBT = {};
	OptixShaderBindingTable shadowSBT = {};

	//inter::vector<OptixProgramGroup>;
};

KRR_NAMESPACE_END