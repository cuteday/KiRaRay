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
#include <functional>

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
	void renderPasses(bool annotate = false);
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
	struct BenchmarkResult {
		std::vector<float> image;
		json timings;
	};

	HeadlessRenderer(const json &config, const fs::path &assetRoot = {}, bool validation = true);
	std::vector<float> render(int64_t frames, uint64_t seed = 0);
	BenchmarkResult benchmark(int64_t frames, int64_t warmup = 8, uint64_t seed = 0,
		bool capture = false, const std::function<void()> &onCaptureBegin = {},
		const std::function<void()> &onCaptureEnd = {});

private:
	BenchmarkResult renderBatch(int64_t frames, int64_t warmup, uint64_t seed,
		bool measure, bool capture, const std::function<void()> &onCaptureBegin = {},
		const std::function<void()> &onCaptureEnd = {});
	bool mBatchActive{};
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
