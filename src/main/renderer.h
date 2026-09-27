#pragma once

#include "graphics/device.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "graphics/uirender.h"
#include "device/buffer.h"
#include "device/context.h"
#include "scene/importer.h"
#include "render/profiler/ui.h"
#include "render/profiler/fps.h"
#include <functional>
#include <optional>
#include <atomic>
#include <thread>

NAMESPACE_BEGIN(krr)

class GBufferPass;

class Renderer : public DeviceManager {
public:
	Renderer();
	~Renderer() override;
	Renderer(const Renderer &) = delete;
	Renderer &operator=(const Renderer &) = delete;

	void setScene(Scene::SharedPtr scene);
	void loadConfigFrom(fs::path path);
	void loadConfig(const json &config, Scene::SharedPtr scene = {});
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

struct CameraState {
	Matrix4f cameraToWorld{Matrix4f::Identity()};
	Matrix4f projection{Matrix4f::Identity()};
	float nearClip{-1.f};
	bool orthographic{};
};

class RenderSession : public Renderer {
public:
	struct DepthSnapshot {
		std::vector<float> linear, projected;
	};
	struct Snapshot {
		std::vector<float> image;
		Vector2i size;
		uint64_t completedFrames{};
		uint64_t generation{};
		std::shared_ptr<const DepthSnapshot> depth;
	};
	struct SnapshotRequest {
		Snapshot snapshot;
		RenderContext::ReadbackHandle readback;
	};

	RenderSession(const json &config, const fs::path &assetRoot = {}, bool validation = true);
	~RenderSession() override;
	void initialize(uint64_t seed = 0, Scene::SharedPtr scene = {});
	uint32_t step(int64_t frames = 1);
	void wait();
	Snapshot snapshot(bool depth = false);
	// Completing a readback does not wait for later rendering or permit managed scene access.
	std::optional<SnapshotRequest> requestSnapshot(bool depth = false);
	bool isSnapshotReady(const SnapshotRequest &request);
	Snapshot collectSnapshot(SnapshotRequest &request, bool wait = false);
	void reset(uint64_t seed = 0);
	void updateCamera(const CameraState &camera);
	void resize(const Vector2i &size);
	void replaceScene(Scene::SharedPtr scene);
	void finish();
	void close() noexcept override;
	void requestCancel() noexcept { mCancelRequested = true; }
	uint64_t getCompletedFrames() const { return mCompletedFrames; }
	uint64_t getGeneration() const { return mGeneration; }
	uint64_t getDepthCaptureCount() const { return mDepthCaptureCount; }
	Scene::SharedPtr getScene() const { return mScene; }

protected:
	void renderFrame(bool annotate = false);
	void releaseScene();
	void synchronize();
	void requireReady() const;
	void requireOwner() const;

private:
	void updateScene();
	std::shared_ptr<const DepthSnapshot> captureDepth();
	const std::thread::id mOwnerThread{std::this_thread::get_id()};
	std::atomic<bool> mCancelRequested{};
	uint64_t mCompletedFrames{}, mUpdateSerial{}, mGeneration{};
	bool mInitialized{}, mFinished{};
	std::unique_ptr<GBufferPass> mDepthPass;
	std::shared_ptr<const DepthSnapshot> mDepth;
	uint64_t mDepthCaptureCount{};
};

class HeadlessRenderer : public RenderSession {
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
	void onWindowFocus(int focused) override;

	void run();
	void renderUI();

	void captureFrame(bool hdr = false, fs::path filename = "");
	
	void saveConfig(string path);

private:
	UIRenderer::SharedPtr mpUIRenderer;
	ProfilerUI::UniquePtr mProfilerUI;
};

NAMESPACE_END(krr)
