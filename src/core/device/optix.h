#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#include "raytracing.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	HitgroupSBTData data;
};

class OptiXBackendInterface {
public:
	OptiXBackendInterface() = default;

	static OptixModule createOptiXModule(OptixDeviceContext optixContext, const char* ptx);
	static OptixPipelineCompileOptions getPipelineCompileOptions();

	static OptixProgramGroup createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule,
		const char* closest, const char* any, const char* intersect);

	static OptixTraversableHandle buildASFromInputs(OptixDeviceContext optixContext, 
		CUstream cudaStream, const std::vector<OptixBuildInput> &buildInputs, CUDABuffer& accelBuffer); 
	static OptixTraversableHandle buildTriangleMeshGAS(OptixDeviceContext optixContext,
													   CUstream cudaStream,
													   const rt::MeshData &mesh,
													   CUDABuffer &accelBuffer);
};

struct OptiXInitializeParameters {
	char* ptx;
	std::vector<string> raygenEntries;
	std::vector<string> rayTypes;
	std::vector<std::tuple<bool, bool, bool>> rayClosestShaders;

	OptiXInitializeParameters& setPTX(char* ptx) {
		this->ptx = ptx;
		return *this;
	}
	OptiXInitializeParameters &addRaygenEntry(string entryName) {
		raygenEntries.push_back(entryName);
		return *this;
	}
	OptiXInitializeParameters &addRayType(string typeName, bool closestHit,
										  bool anyHit, bool intersect) {
		rayTypes.push_back(typeName);
		rayClosestShaders.push_back({closestHit, anyHit, intersect});
		return *this;
	}
};

class OptiXBackend : public OptiXBackendInterface {
public:
	OptiXBackend() = default;
	
	void initialize(const OptiXInitializeParameters& params);
	void setScene(Scene::SharedPtr scene);
	template <typename T>
	void launch(const T& parameters, string entryPoint, int width,
		int height, int depth = 1) {
		if (height * width * depth == 0) return;
		static T *launchParams{nullptr};
		if (!launchParams) cudaMalloc(&launchParams, sizeof(T));
		if (!entryPoints.count(entryPoint))
			Log(Fatal, "The entrypoint %s is not initialized!", entryPoint.c_str());
		cudaMemcpy(launchParams, &parameters, sizeof(T),
				   cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, CUdeviceptr(launchParams), 
			sizeof(T), &SBT[entryPoints[entryPoint]], width, height, depth));
	}

	OptixTraversableHandle getRootTraversable() const { return traversableIAS; }
	rt::SceneData getSceneData() const { return scene->mpSceneRT->getSceneData(); }

protected:
	void buildAccelStructure(Scene::SharedPtr scene);
	void buildShaderBindingTable(Scene::SharedPtr scene);

	OptixProgramGroup createRaygenPG(const char *entrypoint) const;
	OptixProgramGroup createMissPG(const char *entrypoint) const;
	OptixProgramGroup createIntersectionPG(const char *closest, const char *any,
										   const char *intersect) const;

	OptixModule optixModule;
	OptixPipeline optixPipeline;
	OptixDeviceContext optixContext;
	CUstream cudaStream;

	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs; 
	inter::vector<RaygenRecord> raygenRecords;
	inter::vector<HitgroupRecord> hitgroupRecords;
	inter::vector<MissRecord> missRecords;

	inter::vector<OptixInstance> instancesIAS;
	std::vector<OptixTraversableHandle> traversablesGAS;
	std::vector<CUDABuffer> accelBuffersGAS;
	CUDABuffer accelBufferIAS;
	OptixTraversableHandle traversableIAS{};

	std::map<string, int> entryPoints;
	std::vector<OptixShaderBindingTable> SBT;
	
	Scene::SharedPtr scene;
	OptiXInitializeParameters optixParameters;
};


KRR_NAMESPACE_END