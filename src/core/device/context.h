#pragma once

#include "common.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "interop.h"
#include "renderpass.h"

KRR_NAMESPACE_BEGIN

class Context{
public:
	using SharedPtr = std::shared_ptr<Context>;

	Context() { initialize(); }
	~Context() { finalize(); }

	void initialize();
	void finalize();

	CUcontext cudaContext;
	CUstream cudaStream{ 0 };
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext;
	Allocator* alloc;
};

extern Context::SharedPtr gpContext;

KRR_NAMESPACE_END