#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include <optional>

#include "raytracing.h"
#include "context.h"
#include "scene.h"

NAMESPACE_BEGIN(krr)

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
	OptixScene(std::shared_ptr<Scene> scene, const OptixSceneParameters &config = {}) : 
		scene(scene), config(config) {}
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
	OptixSceneParameters config;
};

class OptixSceneSingleLevel : public OptixScene {
public:
	using SharedPtr = std::shared_ptr<OptixSceneSingleLevel>;
	OptixSceneSingleLevel(std::shared_ptr<Scene> scene, const OptixSceneParameters &config = {});

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
	struct MotionKeyframes {
		std::vector<OptixSRTData> keyframes;
		float startTime{0}, endTime{0};
	};

	struct InstanceBuildInput {
		InstanceBuildInput() = default;
		~InstanceBuildInput();

		CUDABuffer accelBuffer;
		gpu::vector<OptixInstance> instances;
		OptixTraversableHandle traversable;

		/* [optional] used if this node has a motion transform, and motion blur is enabled. */
		CUDABuffer transformBuffer;
		OptixTraversableHandle transformTraversable;

		/* children instaces that this node references. */
		std::vector<SceneGraphNode*> nodes;
	};

	using SharedPtr = std::shared_ptr<OptixSceneMultiLevel>;
	OptixSceneMultiLevel(std::shared_ptr<Scene> scene, const OptixSceneParameters &config = {});
	

	OptixTraversableHandle getRootTraversable() const override { return traversableIAS; }
	std::vector<std::weak_ptr<MeshInstance>> getReferencedMeshes() const override { return referencedMeshes; }

protected:
	void buildAccelStructure() override;  // build single-level accel structure
	void updateAccelStructure() override; // update single-level accel structure
private:
	std::optional<MotionKeyframes> getMotionKeyframes(SceneGraphNode* node);
	std::pair<OptixTraversableHandle, int> buildIASForNode(SceneGraphNode* node, std::optional<MotionKeyframes> motion);
	std::vector<std::weak_ptr<MeshInstance>> referencedMeshes;
	std::vector<std::shared_ptr<InstanceBuildInput>> instanceBuildInputs;
	std::vector<CUDABuffer> accelBuffersGAS;
	std::vector<OptixTraversableHandle> traversablesGAS;

	OptixTraversableHandle traversableIAS;
};

struct OptixInitializeParameters {
	char* ptx;
	unsigned int maxTraversableDepth{6};
	std::vector<string> raygenEntries;
	std::vector<string> rayTypes;
	std::vector<OptixModuleCompileBoundValueEntry> boundValues;
	std::vector<std::tuple<bool, bool, bool>> rayClosestShaders;

	OptixInitializeParameters& setMaxTraversableDepth(unsigned int depth) {
		maxTraversableDepth = depth;
		return *this;
	}
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
	OptixInitializeParameters& addBoundValue(size_t offset, size_t size, void* data,
		char* annotation) {
		OptixModuleCompileBoundValueEntry entry;
		entry.pipelineParamOffsetInBytes = offset;
		entry.sizeInBytes				 = size;
		entry.boundValuePtr				 = data;
		entry.annotation				 = annotation;
		boundValues.push_back(entry);
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
		int width, int height, int depth = 1, CUstream stream = 0) {
		if (height * width * depth == 0) return;
		static LaunchParameters<Integrator> *launchParams{nullptr};
		if (!launchParams) cudaMalloc(&launchParams, sizeof(LaunchParameters<Integrator>));
		if (!entryPoints.count(entryPoint))
			Log(Fatal, "The entrypoint %s is not initialized!", entryPoint.c_str());
		cudaMemcpyAsync(launchParams, &parameters, sizeof(LaunchParameters<Integrator>),
						cudaMemcpyHostToDevice, stream);
		OPTIX_CHECK(optixLaunch(optixPipeline, stream, CUdeviceptr(launchParams),
								sizeof(LaunchParameters<Integrator>), &SBT[entryPoints[entryPoint]],
								width, height, depth));
	}

	std::shared_ptr<Scene> getScene() const { return scene; }
	std::shared_ptr<OptixScene> getOptixScene() const;
	rt::SceneData getSceneData() const;
	std::vector<string> getRayTypes() const { return optixParameters.rayTypes; }
	std::vector<string> getRaygenEntries() const { return optixParameters.raygenEntries; }
	OptixTraversableHandle getRootTraversable() const;

	void setParameters(const OptixInitializeParameters &params) { optixParameters = params; }
	OptixInitializeParameters getParameters() const { return optixParameters; }

protected:
	void createOptixModule();
	void createOptixPipeline();
	void buildShaderBindingTable();

	OptixProgramGroup createRaygenPG(const char *entrypoint) const;
	OptixProgramGroup createMissPG(const char *entrypoint) const;
	OptixProgramGroup createIntersectionPG(const char *closest, const char *any,
										   const char *intersect) const;

	OptixModule optixModule;
	OptixPipeline optixPipeline;
	OptixDeviceContext optixContext;

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
	static OptixModule createOptixModule(OptixDeviceContext optixContext,
										 const OptixInitializeParameters& params);
	static OptixPipelineCompileOptions getPipelineCompileOptions();

	static OptixProgramGroup createRaygenPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createMissPG(OptixDeviceContext optixContext, OptixModule optixModule, const char* entrypoint);
	static OptixProgramGroup createIntersectionPG(OptixDeviceContext optixContext, OptixModule optixModule,
		const char* closest, const char* any, const char* intersect);
};

NAMESPACE_END(krr)