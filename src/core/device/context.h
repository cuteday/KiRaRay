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

	void setGlobalConfig(const json &config);
	void updateGlobalConfig(const json &config);
	json getGlobalConfig() const;

	void initialize();
	void finalize();

	json globalConfig{};
	CUcontext cudaContext;
	CUstream cudaStream{ 0 };
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext;
	Allocator* alloc;
};

extern Context::SharedPtr gpContext;

KRR_NAMESPACE_END