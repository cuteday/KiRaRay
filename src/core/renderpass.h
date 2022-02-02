#pragma once

#include "common.h"
#include "math/math.h"
#include "scene.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

class RenderPass{
public:
	KRR_CLASS_DEFINE(RenderPass);

	virtual void resize(vec2i size) { mFrameSize = size; }
	virtual void setEnable(bool enable) { mEnable = enable; }
	virtual void setScene(Scene::SharedPtr scene) { mpScene = scene; }
	virtual Scene::SharedPtr getScene() { return mpScene; }
	virtual void renderUI(){}

	virtual bool onMouseEvent(const io::MouseEvent& mouseEvent) { return false; }
	virtual bool onKeyEvent(const io::KeyboardEvent& keyEvent) { return false; }

protected:
	bool mEnable = true;
	vec2i mFrameSize;
	Scene::SharedPtr mpScene = nullptr;
};

KRR_NAMESPACE_END