#include <cstdlib>

#include "logger.h"
#include "context.h"
#include "renderpass.h"
#include "render/color.h"
#include "render/spectrum.h"

NAMESPACE_BEGIN(krr)

using namespace gpu;
CUDATrackedMemory CUDATrackedMemory::singleton;
std::unique_ptr<Context> gpContext;
std::shared_ptr<RenderPassFactory::map_type> RenderPassFactory::map = nullptr;
std::shared_ptr<RenderPassFactory::configured_map_type> RenderPassFactory::configured_map = nullptr;

namespace {
	static void optixContextLogCallback(unsigned int level,
		const char* tag,
		const char* message,
		void*){
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
	}
}

Context::Context() {
	try {
		initialize();
	} catch (...) {
		finalize();
		throw;
	}
}

Context &Context::ensureInitialized() {
	if (!gpContext) gpContext = std::make_unique<Context>();
	return *gpContext;
}

void Context::initialize() {
	logInfo("Initializing device context");

	// initialize optix and cuda 
	int numDevices{};
	CUDA_CHECK(cudaGetDeviceCount(&numDevices));
	if (numDevices == 0)
		throw std::runtime_error("No CUDA capable devices found!");
	logInfo("Found " + to_string(numDevices) + " CUDA device(s).");
	OPTIX_CHECK(optixInit());

	// set up context
	int deviceID{};
	CUDA_CHECK(cudaGetDevice(&deviceID));
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaStreamCreate(&cudaStream));
	
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, deviceID));
	Log(Success, "KiRaRay is running on " + string(deviceProps.name));
	if (!deviceProps.concurrentManagedAccess)
		Log(Debug, "Concurrent access of managed memory is not supported.");

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		throw std::runtime_error("Error querying current CUDA context: " + std::to_string(cuRes));

	OptixDeviceContextOptions optixContextOptions = {};
	//optixContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &optixContextOptions, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, optixContextLogCallback, nullptr, 4));
	//OPTIX_CHECK(optixDeviceContextSetCacheEnabled(optixContext, false));

	// tracked cuda device memory management
	set_default_resource(&CUDATrackedMemory::singleton);
	alloc = std::make_unique<Allocator>(&CUDATrackedMemory::singleton);

	// initialize spectral rendering resources
	spec::init(*alloc);
#if KRR_RENDER_SPECTRAL
	RGBToSpectrumTable::init(*alloc);
	RGBColorSpace::init(*alloc);
	CUDA_SYNC_CHECK();
#endif
}

void Context::finalize() noexcept {
	if (cudaStream) cudaStreamSynchronize(cudaStream);
	if (optixContext) optixDeviceContextDestroy(optixContext);
	optixContext = nullptr;
	if (cudaStream) cudaStreamDestroy(cudaStream);
	cudaStream = nullptr;
	if (alloc) {
		try {
			CUDATrackedMemory::singleton.release();
		} catch (...) {}
	}
	alloc.reset();
	defaultVkDevice = nullptr;
	cudaContext = nullptr;
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

NAMESPACE_END(krr)
