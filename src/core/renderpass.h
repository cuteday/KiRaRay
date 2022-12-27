#pragma once
#pragma init_seg(lib)

#include "common.h"
#include "window.h"
#include "logger.h"

#include "device/buffer.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class RenderResource {
	RenderResource(string name) :name(name) { glGenBuffers(1, &pbo); }
	~RenderResource() { glDeleteBuffers(1, &pbo); }
	void resize(size_t size) {}
	void registerCUDA();
	void *map() { return nullptr; }
	void unmap() {};
private:
	string name;
	GLuint pbo;
};

class RenderData {
public:
	RenderData() = default;
	RenderResource* getResource(string name);
	RenderResource* createResource(string name);
private:
	std::map<string, RenderResource*> resources;
};

class RenderPass{
public:
	using SharedPtr = std::shared_ptr<RenderPass>;

	RenderPass() = default;
	// Whether this pass is enabled by default
	RenderPass(bool enable) : mEnable(enable) {}

	virtual void resize(const Vector2i& size) { mFrameSize = size; }
	virtual void setEnable(bool enable) { mEnable = enable; }
	virtual void setScene(Scene::SharedPtr scene) { mpScene = scene; }
	virtual Scene::SharedPtr getScene() { return mpScene; }
	virtual void renderUI() {}
	virtual void beginFrame(CUDABuffer& frame) {}
	virtual void render(CUDABuffer& frame) {}
	virtual void endFrame(CUDABuffer& frame) {}

	virtual bool onMouseEvent(const io::MouseEvent& mouseEvent) { return false; }
	virtual bool onKeyEvent(const io::KeyboardEvent& keyEvent) { return false; }

	virtual string getName() const { return "RenderPass"; }
	virtual bool enabled() const { return mEnable; }

protected:
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
	struct exec_register {
		exec_register() = default;
		exec_register(const string &s) { 
			getMap()->insert(std::make_pair(s, &RenderPassFactory::create<T>));
			getConfiguredMap()->insert(std::make_pair(s, &RenderPassFactory::deserialize<T>));
		}
	};
	// will force instantiation of definition of static member
	static exec_register register_object;
	template <typename T, T> struct value {};
	typedef value<exec_register &, register_object> value_user;
	static_assert(&register_object);
};

#define KRR_REGISTER_PASS_DEC(name)                                                                     \
	static RenderPassRegister<name> reg;

#define KRR_REGISTER_PASS_DEF(name)																		\
	RenderPassRegister<name> name::reg(#name);															\
	typename RenderPassRegister<name>::exec_register RenderPassRegister<name>::register_object(#name)                                                   


KRR_NAMESPACE_END