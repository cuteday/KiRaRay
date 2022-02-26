#pragma once

#include "common.h"
#include "optix.h"
#include "device/buffer.h"
#include "renderpass.h"

KRR_NAMESPACE_BEGIN

class Context{
public:
	using SharedPtr = std::shared_ptr<Context>;

	Context() { initialize(); }
	~Context() { finalize(); }

	void initialize();
	void finalize();

//private:

	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext;
};

extern Context::SharedPtr gpContext;

KRR_NAMESPACE_END