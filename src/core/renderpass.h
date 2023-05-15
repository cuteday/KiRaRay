#pragma once
#pragma init_seg(lib)
#include <nvrhi/vulkan.h>
#include <vulkan/cuvk.h>

#include "common.h"
#include "logger.h"

#include "device/buffer.h"
#include "device/cuda.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

namespace vkrhi { using namespace nvrhi; }

class DeviceManager;

class RenderFrame {
public:
	using SharedPtr = std::shared_ptr<RenderFrame>;

	RenderFrame(vkrhi::FramebufferHandle framebuffer) :
		mFramebuffer(framebuffer) {}

	~RenderFrame() { 
		if (mDevice) {	// this RenderFrame is once initialized.
			vk::Device device = mDevice->getNativeObject(nvrhi::ObjectTypes::VK_Device);
			device.waitIdle();
			device.destroySemaphore(mSemaphore.vk_sem);
		}
	}

	operator vkrhi::FramebufferHandle() const { return mFramebuffer; }
	operator vkrhi::IFramebuffer *() const { return mFramebuffer.Get(); }

	vkrhi::FramebufferHandle getFramebuffer() const { return mFramebuffer; }
	
	void getSize(uint32_t& width, uint32_t& height) const {
		auto &textureDesc =
			mFramebuffer->getDesc().colorAttachments[0].texture->getDesc();
		width = textureDesc.width;
		height = textureDesc.height;
	}
	
	CudaRenderTarget getCudaRenderTarget() const {
		uint32_t width, height;
		getSize(width, height);
		return CudaRenderTarget(mCudaFrame, width, height);		
	}

	void initialize(vkrhi::vulkan::IDevice *device) {
		auto cuFriend = std::make_unique<vkrhi::CuVkHandler>(device);
		mDevice		  = device;
		mCudaFrame = cuFriend->mapVulkanTextureToCudaSurface(
			mFramebuffer->getDesc().colorAttachments[0].texture,
			cudaArrayColorAttachment);
		mSemaphore = cuFriend->createCuVkSemaphore(true);
	}

	void vulkanUpdateCuda(cudaExternalSemaphore_t cudaSem, CUstream stream = 0) {
		auto cuFriend = std::make_unique<vkrhi::CuVkHandler>(mDevice);
		auto *device	   = dynamic_cast<vkrhi::vulkan::Device *>(mDevice);
		uint64_t waitValue = device->getQueue(vkrhi::CommandQueue::Graphics)
								->getLastSubmittedID();
		cudaExternalSemaphore_t waitSemaphores[] = {cudaSem};
		cuFriend->cudaWaitExternalSemaphore(stream, waitValue, waitSemaphores);
	}

	void cudaUpdateVulkan(CUstream stream = 0) {
		auto cuFriend = std::make_unique<vkrhi::CuVkHandler>(mDevice);
		cuFriend->cudaSignalExternalSemaphore(stream, ++mSemaphoreValue,
											  &mSemaphore.cuda());
		mDevice->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics,
									   mSemaphore, mSemaphoreValue);
	}

	vkrhi::vulkan::IDevice *mDevice;
	vkrhi::FramebufferHandle mFramebuffer;
	cudaSurfaceObject_t mCudaFrame{};
	uint64_t mSemaphoreValue{};
	vkrhi::CuVkSemaphore mSemaphore{};
};

class RenderPass{
private:
	DeviceManager *mDeviceManager{};
	
public:
	using SharedPtr = std::shared_ptr<RenderPass>;

	RenderPass() = default;
	virtual ~RenderPass() = default;
	RenderPass(DeviceManager *device) : mDeviceManager(device) {}

	virtual void resizing() {}
	virtual void resize(const Vector2i& size) { mFrameSize = size; }
	
	virtual void setEnable(bool enable) { mEnable = enable; }
	virtual void setScene(Scene::SharedPtr scene) { mpScene = scene; }
	virtual Scene::SharedPtr getScene() { return mpScene; }
	virtual void setDeviceManager(DeviceManager *deviceManager) {
		mDeviceManager = deviceManager;
	}

	virtual void tick(float elapsedSeconds) {}
	virtual void beginFrame() {}
	virtual void render(RenderFrame::SharedPtr frame) {}
	virtual void renderUI() {}
	virtual void endFrame() {}
	
	virtual void initialize() {}
	virtual void finalize() {}

	virtual void onWindowClose() {}
	virtual void onWindowIconify(int iconified) {}
	virtual void onWindowFocus(int focused) {}
	virtual void onWindowRefresh() {}
	virtual void onWindowPosUpdate(int xpos, int ypos) {}
	virtual bool onMouseEvent(const io::MouseEvent& mouseEvent) { return false; }
	virtual bool onKeyEvent(const io::KeyboardEvent& keyEvent) { return false; }

	// Is this render pass contains any cuda/vulkan-based operations? 
	// Used mainly for synchronization between these two GAPIs.
	virtual bool isCudaPass() const { return true; }	
	virtual bool isVulkanPass() const { return true; } 

	virtual string getName() const { return "RenderPass"; }
	virtual bool enabled() const { return mEnable; }

protected:
	[[nodiscard]] DeviceManager *getDeviceManager() const; 
	[[nodiscard]] vk::Device getVulkanNativeDevice() const;
	[[nodiscard]] vkrhi::vulkan::IDevice *getVulkanDevice() const;
	[[nodiscard]] size_t getFrameIndex() const;

	friend void to_json(json &j, const RenderPass &p) {
		j = json{ { "enable", p.mEnable } };
	}

	friend void from_json(const json &j, RenderPass &p) {
		j.at("enable").get_to(p.mEnable);
	}
	
	bool mEnable = true;
	Vector2i mFrameSize{};
	Scene::SharedPtr mpScene = nullptr;
};

class RenderPassFactory {
public:
	typedef std::map<string, std::function<RenderPass::SharedPtr(void)> > map_type;
	typedef std::map<string, std::function<RenderPass::SharedPtr(const json&)>> configured_map_type;
	
	template <typename T> 
	static RenderPass::SharedPtr create() { 
		return std::make_shared<T>(); 
	}

	template <typename T> 
	static RenderPass::SharedPtr deserialize(const json &serde) { 
		auto ret = std::make_shared<T>(); 
		*ret	 = serde.get<T>();
		return ret;
	}

	static RenderPass::SharedPtr createInstance(std::string const &s) {
		auto map			  = getMap();
		map_type::iterator it = map->find(s);
		if (it == map->end()) {
			Log(Error, "Could not create instance for %s: check if the pass is registered.", s.c_str());
			return 0;
		}
		return it->second();
	}

	static RenderPass::SharedPtr deserizeInstance(std::string const &s, const json &serde) {
		auto configured_map				 = getConfiguredMap();
		configured_map_type::iterator it = configured_map->find(s);
		if (it == configured_map->end()) {
			Log(Error, "Could not deserialize instance for %s:" 
					"check if the pass is registered, and serde methods implemented.",
				s.c_str());
			return 0;
		}
		return it->second(serde);
	}

protected:
	static std::shared_ptr<map_type> getMap() {
		if (!map) { map.reset(new map_type); }
		return map;
	}

	static std::shared_ptr<configured_map_type> getConfiguredMap() {
		if (!configured_map) {
			configured_map.reset(new configured_map_type);
		}
		return configured_map;
	}

private:
	/* The two map members are initialized in context.cpp */
	static std::shared_ptr<map_type> map;
	static std::shared_ptr<configured_map_type> configured_map;
};

template <typename T> 
class RenderPassRegister : RenderPassFactory {
public:
	RenderPassRegister(const string &s) {
		getMap()->insert(std::make_pair(s, &RenderPassFactory::create<T>));
		getConfiguredMap()->insert(std::make_pair(s, &RenderPassFactory::deserialize<T>));
	}

private:
	RenderPassRegister()  = default; 
};

#define KRR_REGISTER_PASS_DEC(name) static RenderPassRegister<name> reg;

#define KRR_REGISTER_PASS_DEF(name)	RenderPassRegister<name> name::reg(#name);

KRR_NAMESPACE_END