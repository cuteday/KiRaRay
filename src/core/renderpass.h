#pragma once

#include "common.h"
#include "logger.h"
#include "device/optix.h"
#include "graphics/rendercontext.h"

NAMESPACE_BEGIN(krr)

class DeviceManager;

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

	// CUDA-only passes share a handoff; mixed passes manage CudaScope themselves.
	virtual bool isCudaPass() const { return true; }	

	virtual string getName() const { return "RenderPass"; }
	virtual bool enabled() const { return mEnable; }

protected:
	[[nodiscard]] DeviceManager *getDeviceManager() const; 
	[[nodiscard]] nvrhi::IDevice *getDevice() const;
	[[nodiscard]] size_t getFrameIndex() const;
	[[nodiscard]] uint64_t getSeed() const;
	[[nodiscard]] bool isHeadless() const;
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
	static bool isRegistered(const std::string &name) {
		return getMap()->count(name) != 0;
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
