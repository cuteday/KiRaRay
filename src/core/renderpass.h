#pragma once
#include <nvrhi/vulkan.h>
#include <vulkan/cuvk.h>

#include "common.h"
#include "logger.h"

#include "device/buffer.h"
#include "device/cuda.h"
#include "device/optix.h"
#include "scene.h"

NAMESPACE_BEGIN(krr)

namespace vkrhi { using namespace nvrhi; }

class DeviceManager;

class RenderTexture {
public:
	using SharedPtr = std::shared_ptr<RenderTexture>;

	RenderTexture(vkrhi::IDevice *device, vkrhi::TextureHandle texture);
	~RenderTexture();

	static RenderTexture::SharedPtr create(vkrhi::IDevice *device, 
		const Vector2i size, vkrhi::Format format, const std::string name = "");

	operator vkrhi::TextureHandle() const { return mTexture; }
	operator vkrhi::ITexture *() const { return mTexture.Get(); }
	operator CudaRenderTarget() const { return getCudaRenderTarget(); }
	vkrhi::ITexture *getVulkanTexture() const { return mTexture.Get(); }

	Vector2i getSize() const {
		auto &textureDesc = mTexture->getDesc();
		return Vector2i{textureDesc.width, textureDesc.height};
	}

	CudaRenderTarget getCudaRenderTarget() const {
		return CudaRenderTarget{mCudaSurface, getSize()[0], getSize()[1]};
	}

protected:
	static vkrhi::TextureDesc getVulkanDesc(const Vector2i size, vkrhi::Format format,
											const std::string name = "");

	vkrhi::TextureHandle mTexture;
	cudaSurfaceObject_t mCudaSurface;
};

class RenderTarget {
public:
	using SharedPtr = std::shared_ptr<RenderTarget>;
	RenderTarget(vkrhi::IDevice *device) : mDevice(device) {}
	~RenderTarget() = default;

	vkrhi::IFramebuffer *getFramebuffer() const { return mFramebuffer.Get(); }
	RenderTexture *getColorTexture() const { return mColor.get(); }
	RenderTexture *getDepthTexture() const { return mDepth.get(); }
	RenderTexture *getDiffuseTexture() const { return mDiffuse.get(); }
	RenderTexture *getSpecularTexture() const { return mSpecular.get(); }
	RenderTexture *getNormalTexture() const { return mNormal.get(); }
	RenderTexture *getEmissiveTexture() const { return mEmissive.get(); }
	RenderTexture *getMotionTexture() const { return mMotion.get(); }

	void setDepthEnabled(bool enable) { mEnableDepth = enable; resize(mSize); }
	void setDiffuseEnabled(bool enable) { mEnableDiffuse = enable; resize(mSize); }
	void setSpecularEnabled(bool enable) { mEnableSpecular = enable; resize(mSize); }
	void setNormalEnabled(bool enable) { mEnableNormal = enable; resize(mSize); }
	void setEmissiveEnabled(bool enable) { mEnableEmissive = enable; resize(mSize); }
	void setMotionEnabled(bool enable) { mEnableMotion = enable; resize(mSize); }

	void resize(const Vector2i size);
	Vector2i getSize() const { return mSize; }
	bool isUpdateNeeded(const Vector2i size) const { return size != mSize; };

protected:
	vkrhi::FramebufferHandle mFramebuffer;
	RenderTexture::SharedPtr mColor{}; 
	RenderTexture::SharedPtr mDepth{};
	RenderTexture::SharedPtr mDiffuse{};
	RenderTexture::SharedPtr mSpecular{};
	RenderTexture::SharedPtr mNormal{};
	RenderTexture::SharedPtr mEmissive{};
	RenderTexture::SharedPtr mMotion{};

	bool mEnableDepth{};
	bool mEnableDiffuse{};
	bool mEnableSpecular{};
	bool mEnableNormal{};
	bool mEnableEmissive{};
	bool mEnableMotion{};

	Vector2i mSize{};
	vkrhi::IDevice *mDevice{};
};

class RenderContext {
public:
	struct CudaScope {
		CudaScope(RenderContext *_ctx): ctx(_ctx) { ctx->sychronizeVulkan(); }
		~CudaScope() { ctx->sychronizeCuda(); }
		RenderContext* ctx;
	};
	using SharedPtr = std::shared_ptr<RenderContext>;
	RenderContext(nvrhi::IDevice* device);
	~RenderContext();

	nvrhi::IDevice *getDevice() const { return mDevice; }
	nvrhi::ICommandList *getCommandList() const { return mCommandList.Get(); }
	nvrhi::IFramebuffer *getFramebuffer() const { return mRenderTarget->getFramebuffer(); }
	RenderTexture *getColorTexture() const { return mRenderTarget->getColorTexture(); } 
	CUstream getCudaStream() const { return mCudaStream; }
	RenderTarget::SharedPtr getRenderTarget() const { return mRenderTarget; }
	Scene::SharedPtr getScene() const { return mScene; }
	vkrhi::CuVkSemaphore getCudaSemaphore() const { return mCudaSemaphore; }
	vkrhi::CuVkSemaphore getVulkanSemaphore() const { return mVulkanSemaphore; }
	
	void setScene(Scene::SharedPtr scene);
	void resize(const Vector2i size);
	void sychronizeCuda();
	void sychronizeVulkan();
	json *getJson() { return &mJson; }

private: 
	friend class CudaScope;
	friend class DeviceManager;
	nvrhi::IDevice* mDevice;
	Scene::SharedPtr mScene;
	nvrhi::CommandListHandle mCommandList;
	RenderTarget::SharedPtr mRenderTarget;
	std::unique_ptr<vkrhi::CuVkHandler> mCudaHandler;
	vkrhi::CuVkSemaphore mCudaSemaphore;
	vkrhi::CuVkSemaphore mVulkanSemaphore; 
	uint64_t mCudaSemaphoreValue{};
	CUstream mCudaStream;
	json mJson{};
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
	virtual void resize(const Vector2i& size) {}
	
	virtual void setEnable(bool enable) { mEnable = enable; }
	virtual void setScene(Scene::SharedPtr scene) { mScene = scene; }
	virtual Scene::SharedPtr getScene() { return mScene; }
	virtual void setDeviceManager(DeviceManager *deviceManager) {
		mDeviceManager = deviceManager;
	}
	// The total time elapsed after the first frame, in seconds.
	virtual void tick(float elapsedSeconds) {}
	virtual void beginFrame(RenderContext *context) {}
	virtual void render(RenderContext *context) {}
	virtual void endFrame(RenderContext *context) {}
	virtual void renderUI() {}
	
	virtual void initialize() {}
	virtual void finalize() {}

	virtual void onWindowClose() {}
	virtual void onWindowIconify(int iconified) {}
	virtual void onWindowFocus(int focused) {}
	virtual void onWindowRefresh() {}
	virtual void onWindowPosUpdate(int xpos, int ypos) {}
	virtual bool onMouseEvent(const io::MouseEvent& mouseEvent) { return false; }
	virtual bool onKeyEvent(const io::KeyboardEvent& keyEvent) { return false; }

	// Is this render pass contains any cuda operations? 
	// Used mainly for synchronization between these two GAPIs.
	virtual bool isCudaPass() const { return true; }	

	virtual string getName() const { return "RenderPass"; }
	virtual bool enabled() const { return mEnable; }

protected:
	[[nodiscard]] DeviceManager *getDeviceManager() const; 
	[[nodiscard]] vk::Device getVulkanNativeDevice() const;
	[[nodiscard]] vkrhi::vulkan::IDevice *getVulkanDevice() const;
	[[nodiscard]] size_t getFrameIndex() const;
	[[nodiscard]] Vector2i getFrameSize() const;

	friend void to_json(json &j, const RenderPass &p) {
		j = json{ { "enable", p.mEnable } };
	}

	friend void from_json(const json &j, RenderPass &p) {
		j.at("enable").get_to(p.mEnable);
	}
	
	bool mEnable = true;
	Scene::SharedPtr mScene = nullptr;
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

NAMESPACE_END(krr)