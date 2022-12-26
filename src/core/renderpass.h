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
	KRR_CLASS_DEFINE(RenderPass, mEnable);

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
	bool mEnable = true;
	Vector2i mFrameSize;
	Scene::SharedPtr mpScene = nullptr;
};

class RenderPassFactory {
public:
	typedef std::map<string, RenderPass::SharedPtr(*) ()> map_type;
	
	template <typename T> 
	static RenderPass::SharedPtr create() { 
		return RenderPass::SharedPtr(new T()); 
	}

	static RenderPass::SharedPtr createInstance(std::string const &s) {
		map_type::iterator it = getMap()->find(s);
		if (it == getMap()->end()) {
			Log(Error, "Could not create instance for %s: check if the pass is registered.", s.c_str());
			return 0;
		}
		return it->second();
	}

protected:
	static std::shared_ptr<map_type> getMap() {
		if (!map) { map.reset(new map_type); }
		return map;
	}

private:
	static std::shared_ptr<map_type> map;
};

template <typename T> 
class RenderPassRegister : RenderPassFactory {
public:
	RenderPassRegister(const string &s) { 
		getMap()->insert(std::make_pair(s, &RenderPassFactory::create<T>)); 
	}

private:
	struct exec_register {
		exec_register() = default;
		exec_register(const string &s) { 
			getMap()->insert(std::make_pair(s, &RenderPassFactory::create<T>));
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