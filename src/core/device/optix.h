#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#include "raytracing.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class SceneGraphNode;
class MeshInstance;

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

class OptixScene {
public:
	using SharedPtr = std::shared_ptr<OptixScene>;
	OptixScene(std::shared_ptr<Scene> scene) : scene(scene) {}
	virtual ~OptixScene() = default;

	static OptixTraversableHandle buildASFromInputs(OptixDeviceContext optixContext, 
		CUstream cudaStream, const std::vector<OptixBuildInput> &buildInputs, 
		CUDABuffer& accelBuffer, bool compact = false, bool update = false); 
	static OptixTraversableHandle buildTriangleMeshGAS(OptixDeviceContext optixContext,
													   CUstream cudaStream,
													   const rt::MeshData &mesh,
													   CUDABuffer &accelBuffer);

	rt::SceneData getSceneData() const;
	virtual OptixTraversableHandle getRootTraversable() const = 0;
	virtual std::vector<std::weak_ptr<MeshInstance>> getReferencedMeshes() const = 0;
	virtual void update();

protected:
	virtual void buildAccelStructure() = 0;				// build single-level accel structure
	virtual void updateAccelStructure() = 0;			// update single-level accel structure

	std::weak_ptr<Scene> scene;
};

class OptixSceneSingleLevel : public OptixScene {
public:
	using SharedPtr = std::shared_ptr<OptixSceneSingleLevel>;
	OptixSceneSingleLevel(std::shared_ptr<Scene> scene);

	OptixTraversableHandle getRootTraversable() const override { return traversableIAS; }
	std::vector<std::weak_ptr<MeshInstance>> getReferencedMeshes() const override { return referencedMeshes; }

protected:
	void buildAccelStructure() override; // build single-level accel structure
	void updateAccelStructure() override; // update single-level accel structure
private:
	std::vector<std::weak_ptr<MeshInstance>> referencedMeshes;
	gpu::vector<OptixInstance> instancesIAS;
	CUDABuffer accelBufferIAS{};

	std::vector<OptixTraversableHandle> traversablesGAS;
	std::vector<CUDABuffer> accelBuffersGAS;
	OptixTraversableHandle traversableIAS;
};

class OptixSceneMultiLevel : public OptixScene {
public:
	struct InstanceBuildInput {
		InstanceBuildInput() = default;
		~InstanceBuildInput() { accelBuffer.free(); }

		CUDABuffer accelBuffer;
		gpu::vector<OptixInstance> instances;
		std::vector<SceneGraphNode*> nodes;
	};

	using SharedPtr = std::shared_ptr<OptixSceneMultiLevel>;
	OptixSceneMultiLevel(std::shared_ptr<Scene> scene);

	OptixTraversableHandle getRootTraversable() const override { return traversableIAS; }
	std::vector<std::weak_ptr<MeshInstance>> getReferencedMeshes() const override { return referencedMeshes; }

protected:
	void buildAccelStructure() override;  // build single-level accel structure
	void updateAccelStructure() override; // update single-level accel structure
private:
	OptixTraversableHandle buildIASForNode(SceneGraphNode* node);
	std::vector<std::weak_ptr<MeshInstance>> referencedMeshes;
	std::vector<std::shared_ptr<InstanceBuildInput>> instanceBuildInputs;
	std::vector<CUDABuffer> accelBuffersGAS;
	std::vector<OptixTraversableHandle> traversablesGAS;

	OptixTraversableHandle traversableIAS;
};

struct OptixInitializeParameters {
	char* ptx;
	std::vector<string> raygenEntries;
	std::vector<string> rayTypes;
	std::vector<std::tuple<bool, bool, bool>> rayClosestShaders;

	OptixInitializeParameters& setPTX(char* ptx) {
		this->ptx = ptx;
		return *this;
	}
	OptixInitializeParameters &addRaygenEntry(string entryName) {
		raygenEntries.push_back(entryName);
		return *this;
	}
	OptixInitializeParameters &addRayType(string typeName, bool closestHit,
										  bool anyHit, bool intersect) {
		rayTypes.push_back(typeName);
		rayClosestShaders.push_back({closestHit, anyHit, intersect});
		return *this;
	}
};

class OptixBackend {
public:
	using SharedPtr = std::shared_ptr<OptixBackend>;
	OptixBackend()	= default; 
	~OptixBackend() = default;

	void initialize(const OptixInitializeParameters& params);
	void setScene(std::shared_ptr<Scene> _scene);
	template <typename Integrator>
	void launch(const LaunchParameters<Integrator>& parameters, string entryPoint, 
		int width, int height, int depth = 1) {
		if (height * width * depth == 0) return;
		static LaunchParameters<Integrator> *launchParams{nullptr};
		if (!launchParams) cudaMalloc(&launchParams, sizeof(LaunchParameters<Integrator>));
		if (!entryPoints.count(entryPoint))
			Log(Fatal, "The entrypoint %s is not initialized!", entryPoint.c_str());
		cudaMemcpyAsync(launchParams, &parameters, sizeof(LaunchParameters<Integrator>),
						cudaMemcpyHostToDevice, cudaStream);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, CUdeviceptr(launchParams),
								sizeof(LaunchParameters<Integrator>), &SBT[entryPoints[entryPoint]],
								width, height, depth));
	}

	std::shared_ptr<Scene> getScene() const { return scene; }
	std::shared_ptr<OptixScene> getOptixScene() const;
	rt::SceneData getSceneData() const;
	std::vector<string> getRayTypes() const { return optixParameters.rayTypes; }
	std::vector<string> getRaygenEntries() const { return optixParameters.raygenEntries; }
	OptixTraversableHandle getRootTraversable() const;

protected:
	void buildShaderBindingTable();

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
	gpu::vector<RaygenRecord> raygenRecords;
	gpu::vector<HitgroupRecord> hitgroupRecords;
	gpu::vector<MissRecord> missRecords;

	std::map<string, int> entryPoints;
	std::vector<OptixShaderBindingTable> SBT;
	
	std::shared_ptr<Scene> scene;
	OptixInitializeParameters optixParameters;

public:
	static const size_t OPTIX_MAX_RAY_TYPES = 3;	// Radiance, Shadow, ShadowTransmission
	static OptixModule createOptixModule(OptixDeviceContext optixContext, const char* ptx);
	static OptixPipelineCompileOptions getPipelineCompileOptions();

	static OptixProgramGroup createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule,
		const char* closest, const char* any, const char* intersect);
};

KRR_NAMESPACE_END