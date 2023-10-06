#include <cstdlib>

#include "logger.h"
#include "context.h"
#include "renderpass.h"

KRR_NAMESPACE_BEGIN

using namespace inter;
Context::SharedPtr gpContext;
std::shared_ptr<RenderPassFactory::map_type> RenderPassFactory::map = nullptr;
std::shared_ptr<RenderPassFactory::configured_map_type> RenderPassFactory::configured_map = nullptr;
CUDATrackedMemory CUDATrackedMemory::singleton;

namespace {
	static void optixContextLogCallback(unsigned int level,
		const char* tag,
		const char* message,
		void*){
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
	}
}

void Context::initialize() {
	logInfo("Initializing device context");

	// initialize optix and cuda 
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		logFatal("No CUDA capable devices found!");
	logInfo("Found " + to_string(numDevices) + " CUDA device(s).");
	OPTIX_CHECK(optixInit());

	// set up context
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	//CUDA_CHECK(cudaStreamCreate(&cudaStream));
	cudaStream = 0;

	cudaGetDeviceProperties(&deviceProps, deviceID);
	Log(Success, "KiRaRay is running on " + string(deviceProps.name));
	if (!deviceProps.concurrentManagedAccess)
		Log(Debug, "Concurrent access of managed memory is not supported.");

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		logError("Error querying current context: error code " + cuRes);

	OptixDeviceContextOptions optixContextOptions = {};
	//optixContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &optixContextOptions, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, optixContextLogCallback, nullptr, 4));
	//OPTIX_CHECK(optixDeviceContextSetCacheEnabled(optixContext, false));

	// tracked cuda device memory management
	set_default_resource(&CUDATrackedMemory::singleton);
	alloc = new Allocator(&CUDATrackedMemory::singleton);
}

void Context::finalize(){
	CUDA_SYNC_CHECK();
	optixDeviceContextDestroy(optixContext);
	delete alloc;
	cuCtxDestroy(cudaContext);
}

void Context::terminate() { 
	finalize(); 
	abort();
}

void Context::setGlobalConfig(const json &config) { globalConfig = config; }

void Context::setDefaultVkDevice(nvrhi::vulkan::IDevice *device) { defaultVkDevice = device; }

json Context::getGlobalConfig() const { return globalConfig; }

nvrhi::vulkan::IDevice *Context::getDefaultVkDevice() const { return defaultVkDevice; }

void Context::updateGlobalConfig(const json &config) { globalConfig.update(config); }

KRR_NAMESPACE_END