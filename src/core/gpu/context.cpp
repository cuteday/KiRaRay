#include "logger.h"
#include "context.h"

KRR_NAMESPACE_BEGIN

Context::SharedPtr gpContext;

namespace {
	static void optixContextLogCallback(unsigned int level,
		const char* tag,
		const char* message,
		void*){
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
	}
}

void Context::initialize() {
	// initialize optix and cuda 
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		logFatal("#krr: no CUDA capable devices found!");
	logInfo("Found " + to_string(numDevices) + " CUDA devices!");
	OPTIX_CHECK(optixInit());

	// set up context
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&cudaStream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	logInfo("#krr: running on device: " + string(deviceProps.name));

	CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		logError("Error querying current context: error code " + cuRes);

	OptixDeviceContextOptions optixContextOptions = {};
	//optixContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &optixContextOptions, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, optixContextLogCallback, nullptr, 4));
}

void Context::finalize(){
	optixDeviceContextDestroy(optixContext);
	cuCtxDestroy(cudaContext);
}

KRR_NAMESPACE_END