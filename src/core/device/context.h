#pragma once

#include "common.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "device/gpustd.h"
#include <optix.h>

NAMESPACE_BEGIN(krr)

class Context{
public:
	using SharedPtr = std::shared_ptr<Context>;

	Context();
	~Context() { finalize(); }
	static Context &ensureInitialized();
	void resetState() { globalConfig = json::object(); exit = false; }

	void setGlobalConfig(const json &config);
	void updateGlobalConfig(const json &config);
	json getGlobalConfig() const;
	void requestExit() { exit = true; }
	bool shouldQuit() const { return exit; };

	void initialize();
	void finalize() noexcept;
	void terminate();

	json globalConfig{};
	CUcontext cudaContext{};
	CUstream cudaStream{ 0 };
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext{};
	std::unique_ptr<Allocator> alloc;
	// signal bits
	bool exit{};
};

extern std::unique_ptr<Context> gpContext;

#define KRR_DEFAULT_STREAM gpContext->cudaStream

NAMESPACE_END(krr)
