#pragma once

#include "common.h"
#include "device/buffer.h"
#include "device/memory.h"
#include "nvrhi/vulkan.h"
#include "nvrhi/vulkan/vulkan-backend.h"
#include "device/gpustd.h"
#include "renderpass.h"

NAMESPACE_BEGIN(krr)

class Context{
public:
	using SharedPtr = std::shared_ptr<Context>;

	Context() { initialize(); }
	~Context() { finalize(); }

	void setGlobalConfig(const json &config);
	void setDefaultVkDevice(nvrhi::vulkan::IDevice *device);
	void updateGlobalConfig(const json &config);
	json getGlobalConfig() const;
	nvrhi::vulkan::IDevice *getDefaultVkDevice() const;
	void requestExit() { exit = true; }
	bool shouldQuit() const { return exit; };

	void initialize();
	void finalize();
	void terminate();

	json globalConfig{};
	CUcontext cudaContext;
	CUstream cudaStream{ 0 };
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext;
	nvrhi::vulkan::IDevice *defaultVkDevice{};
	Allocator* alloc;
	// signal bits
	bool exit{};
};

extern std::unique_ptr<Context> gpContext;

NAMESPACE_END(krr)