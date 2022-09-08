#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "device/buffer.h"
#include "device/context.h"
#include "scene/importer.h"
#include "render/profiler/ui.h"
#include "render/profiler/fps.h"

KRR_NAMESPACE_BEGIN

class RenderApp : public WindowApp{
public:
	RenderApp(const char title[], Vector2i size = { 1280, 720 })
		: WindowApp(title, size, true, false) {}
	RenderApp(const char title[], Vector2i size, std::vector<RenderPass::SharedPtr> passes)
		: WindowApp(title, size, true, false), mpPasses(passes) {}

	void resize(const Vector2i size) override;

	// Process signals passed down from direct imgui callback (imgui do not capture it)
	virtual void onMouseEvent(io::MouseEvent &mouseEvent) override;
	virtual void onKeyEvent(io::KeyboardEvent &keyEvent) override;

	void setScene(Scene::SharedPtr scene);
	void setPasses(const std::vector<RenderPass::SharedPtr> passes) { mpPasses = passes; }

	void render() override;
	void run() override;
	void renderUI() override;
	void draw() override;

	void captureFrame(bool hdr = false, fs::path filename = "");
	void saveConfig(string path);
	void loadConfig(fs::path path);

private:
	bool mShowUI{ true };
	bool mPaused{ false };
	int mFrameCount{ 0 };
	int mSpp{ 0 };			// Samples needed tobe rendered, 0 means unlimited.
	FrameRate mFrameRate;
	std::vector<RenderPass::SharedPtr> mpPasses;
	Scene::SharedPtr mpScene;
	ProfilerUI::UniquePtr mpProfilerUI;
	json mConfig{};
	string mConfigPath{};
};

KRR_NAMESPACE_END