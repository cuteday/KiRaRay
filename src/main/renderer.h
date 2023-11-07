#pragma once

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "vulkan/uirender.h"
#include "device/buffer.h"
#include "device/context.h"
#include "scene/importer.h"
#include "render/profiler/ui.h"
#include "render/profiler/fps.h"

KRR_NAMESPACE_BEGIN

class RenderApp : public DeviceManager{
public:
	RenderApp();
	virtual ~RenderApp() = default;

	void backBufferResizing() override;
	void backBufferResized() override;
	void render() override;
	void tick(double elapsedTime /*delta time*/) override;

	void initialize();
	void finalize();

	virtual bool onMouseEvent(io::MouseEvent &mouseEvent) override;
	virtual bool onKeyEvent(io::KeyboardEvent &keyEvent) override;

	void setScene(Scene::SharedPtr scene);
	
	void run();
	void renderUI();

	void captureFrame(bool hdr = false, fs::path filename = "");
	
	void saveConfig(string path);
	void loadConfigFrom(fs::path path);
	void loadConfig(const json config);

private:
	Scene::SharedPtr mScene;
	UIRenderer::SharedPtr mpUIRenderer;
	ProfilerUI::UniquePtr mProfilerUI;
	json mConfig{};
	string mConfigPath{};
};

KRR_NAMESPACE_END