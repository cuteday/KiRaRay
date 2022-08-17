#pragma once

#include "common.h"
#include "window.h"
#include "math/math.h"
#include "device/buffer.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class RenderResource {
	RenderResource(string name) :name(name) { glGenBuffers(1, &pbo); }
	~RenderResource() { glDeleteBuffers(1, &pbo); }
	void resize(size_t size) {}
	void registerCUDA();
	void* map() {};
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

	virtual string getName() const { return ""; }

protected:
	bool mEnable = true;
	Vector2i mFrameSize;
	Scene::SharedPtr mpScene = nullptr;
};

KRR_NAMESPACE_END