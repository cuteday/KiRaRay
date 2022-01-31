#pragma once

#include "common.h"
#include "math/math.h"
#include "scene.h"

KRR_NAMESPACE_BEGIN

class RenderPass{
public:
	KRR_CLASS_DEFINE(RenderPass);

	virtual void resize(vec2i size) { mFrameSize = size; }
	virtual void setEnable(bool enable) { mEnable = enable; }
	virtual void setScene(Scene::SharedPtr scene) { mpScene = scene; }
	virtual void renderUI(){}

protected:
	bool mEnable = true;
	vec2i mFrameSize;
	Scene::SharedPtr mpScene = nullptr;
};

KRR_NAMESPACE_END