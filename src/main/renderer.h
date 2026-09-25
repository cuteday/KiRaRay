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

NAMESPACE_BEGIN(krr)

class Renderer : public DeviceManager {
public:
	Renderer();
	~Renderer() override;
	Renderer(const Renderer &) = delete;
	Renderer &operator=(const Renderer &) = delete;

	void setScene(Scene::SharedPtr scene);
	void loadConfigFrom(fs::path path);
	void loadConfig(const json &config);
	virtual void close() noexcept;
	static void closeActive() noexcept;
	bool isClosed() const { return mClosed; }

protected:
	static void validateConfig(const json &config);
	void initializePasses();
	void renderPasses();
	void clearScene();
	void backBufferResized() override;

	Scene::SharedPtr mScene;
	json mConfig{};
	string mConfigPath{};
	bool mClosed{};

private:
	fs::path mPreviousAssetRoot;
	fs::path mPreviousOutputDir;
};

class HeadlessRenderer : public Renderer {
public:
	HeadlessRenderer(const json &config, const fs::path &assetRoot = {});
	std::vector<float> render(int64_t frames, uint64_t seed = 0);
};

class RenderApp : public Renderer {
public:
	RenderApp();
	~RenderApp() override;

	void backBufferResizing() override;
	void backBufferResized() override;
	void render() override;
	void tick(double elapsedTime /*delta time*/) override;

	void initialize();
	void finalize();
	void close() noexcept override;

	virtual bool onMouseEvent(io::MouseEvent &mouseEvent) override;
	virtual bool onKeyEvent(io::KeyboardEvent &keyEvent) override;

	void run();
	void renderUI();

	void captureFrame(bool hdr = false, fs::path filename = "");
	
	void saveConfig(string path);

private:
	UIRenderer::SharedPtr mpUIRenderer;
	ProfilerUI::UniquePtr mProfilerUI;
};

NAMESPACE_END(krr)
