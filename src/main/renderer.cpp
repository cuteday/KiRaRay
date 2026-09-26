#include "common.h"
#undef NVTX_DISABLE
#include <nvtx3/nvToolsExt.h>
#include "renderer.h"
#include "renderoptions.h"
#include <atomic>
#include <chrono>
#include <cuda_profiler_api.h>

NAMESPACE_BEGIN(krr)

namespace { // status bits
std::atomic<Renderer *> sActiveRenderer{};
static bool sShowUI				 = true;
static bool sSaveHDR			 = false;
static bool sSaveFrames			 = false;
static bool sLockCamera			 = false;
static bool sRequestScreenshot	 = false;
static Vector2ui sCursorPos		 = Vector2ui::Zero();
static size_t sSaveFrameInterval = 2;

class NvtxRange {
public:
	NvtxRange(bool enabled, const char *name) : mEnabled(enabled) {
		if (mEnabled) nvtxRangePushA(name);
	}
	~NvtxRange() { if (mEnabled) nvtxRangePop(); }
private:
	bool mEnabled;
};

class CaptureScope {
public:
	CaptureScope(bool capture, const std::function<void()> &begin,
		const std::function<void()> &end) : mCapture(capture), mBegin(begin), mEnd(end) {}
	~CaptureScope() {
		try { end(); } catch (...) {}
	}
	void begin() {
		if (mCapture) {
			CUDA_CHECK(cudaProfilerStart());
			mProfilerStarted = true;
		}
		if (mBegin) mBegin();
		mCallbackStarted = true;
	}
	void end() {
		std::exception_ptr error;
		if (mCallbackStarted) {
			mCallbackStarted = false;
			try {
				if (mEnd) mEnd();
			} catch (...) { error = std::current_exception(); }
		}
		if (mProfilerStarted) {
			mProfilerStarted = false;
			try {
				CUDA_CHECK(cudaProfilerStop());
			} catch (...) { if (!error) error = std::current_exception(); }
		}
		if (error) std::rethrow_exception(error);
	}
private:
	bool mCapture, mProfilerStarted{}, mCallbackStarted{};
	const std::function<void()> &mBegin, &mEnd;
};

class BatchScope {
public:
	BatchScope(bool &active, bool measure) : mActive(active),
		mProfilerEnabled(Profiler::instance().isEnabled()) {
		mActive = true;
		if (measure) Profiler::instance().setEnabled(false);
	}
	~BatchScope() {
		mActive = false;
		Profiler::instance().setEnabled(mProfilerEnabled);
	}
private:
	bool &mActive;
	bool mProfilerEnabled;
};
}

Renderer::Renderer() : mPreviousAssetRoot(File::cwd()), mPreviousOutputDir(File::outputDir()) {
	Renderer *expected = nullptr;
	if (!sActiveRenderer.compare_exchange_strong(expected, this))
		throw std::runtime_error("Only one renderer may be active at a time");
}

Renderer::~Renderer() { close(); }

void Renderer::closeActive() noexcept {
	if (auto *renderer = sActiveRenderer.load()) renderer->close();
}

void Renderer::validateConfig(const json &config) {
	RenderOptions::validateConfig(config);
	for (const auto &pass : config.at("passes")) {
		const string name = pass.at("name");
		if (!RenderPassFactory::isRegistered(name))
			throw std::invalid_argument("Unknown render pass: " + name);
	}
}

void Renderer::clearScene() {
	std::exception_ptr error;
	try {
		if (gpContext) CUDA_CHECK(cudaStreamSynchronize(gpContext->cudaStream));
		if (mNvrhiDevice) mNvrhiDevice->waitForIdle();
	} catch (...) {
		error = std::current_exception();
	}
	mRenderPasses.clear();
	if (mRenderContext) mRenderContext->setScene(nullptr);
	mScene.reset();
	if (error) std::rethrow_exception(error);
}

void Renderer::close() noexcept {
	if (mClosed) return;
	mClosed = true;
	try {
		clearScene();
	} catch (const std::exception &e) {
		Log(Error, "Scene cleanup: %s", e.what());
	}
	try {
		shutdown();
	} catch (const std::exception &e) {
		Log(Error, "Device cleanup: %s", e.what());
	}
	try {
		File::setCwd(mPreviousAssetRoot);
		File::setOutputDir(mPreviousOutputDir);
	} catch (const std::exception &e) {
		Log(Error, "Restoring renderer paths: %s", e.what());
	}
	sActiveRenderer = nullptr;
}

void Renderer::initializePasses() {
	setScene(mScene);
	for (auto &pass : mRenderPasses) {
		pass->resize(getFrameSize());
		pass->initialize();
	}
	mNvrhiDevice->waitForIdle();
}

void Renderer::renderPasses(bool annotate) {
	for (auto &pass : mRenderPasses) pass->beginFrame(getRenderContext());
	for (auto &pass : mRenderPasses) {
		if (!pass->enabled()) continue;
		NvtxRange range(annotate, annotate ? pass->getName().c_str() : "");
		if (pass->isCudaPass()) getRenderContext()->sychronizeCuda();
		pass->render(getRenderContext());
		if (pass->isCudaPass()) getRenderContext()->sychronizeVulkan();
	}
	for (auto &pass : mRenderPasses) pass->endFrame(getRenderContext());
}

HeadlessRenderer::HeadlessRenderer(const json &config, const fs::path &assetRoot, bool validation) {
	validateConfig(config);
	fs::path root = assetRoot.empty() ? fs::path(KRR_PROJECT_DIR) : fs::absolute(assetRoot);
	if (!fs::is_directory(root))
		throw std::invalid_argument("asset_root must be an existing directory");
	File::setCwd(root);
	mDeviceParams.enableDebugRuntime = validation;
	mDeviceParams.enableNvrhiValidationLayer = validation;
	mConfig = config;
	if (config.contains("resolution")) {
		mDeviceParams.backBufferWidth = config.at("resolution").at(0);
		mDeviceParams.backBufferHeight = config.at("resolution").at(1);
	}
}

std::vector<float> HeadlessRenderer::render(int64_t frames, uint64_t seed) {
	return renderBatch(frames, 0, seed, false, false).image;
}

HeadlessRenderer::BenchmarkResult HeadlessRenderer::benchmark(int64_t frames, int64_t warmup,
	uint64_t seed, bool capture, const std::function<void()> &onCaptureBegin,
	const std::function<void()> &onCaptureEnd) {
	return renderBatch(frames, warmup, seed, true, capture, onCaptureBegin, onCaptureEnd);
}

HeadlessRenderer::BenchmarkResult HeadlessRenderer::renderBatch(int64_t frames, int64_t warmup,
	uint64_t seed, bool measure, bool capture, const std::function<void()> &onCaptureBegin,
	const std::function<void()> &onCaptureEnd) {
	if (mClosed) throw std::runtime_error("Renderer is closed");
	if (mBatchActive) throw std::runtime_error("Renderer is already rendering");
	try {
		using Clock = std::chrono::steady_clock;
		const auto totalStart = Clock::now();
		auto milliseconds = [](auto start, auto end) {
			return std::chrono::duration<double, std::milli>(end - start).count();
		};
		RenderOptions::validateBenchmark(frames, warmup, seed);
		BatchScope batch(mBatchActive, measure);
		BenchmarkResult result;
		auto synchronize = [&] {
			CUDA_CHECK(cudaStreamSynchronize(gpContext->cudaStream));
			mNvrhiDevice->waitForIdle();
		};
		auto renderFrame = [&](int64_t frame) {
			NvtxRange range(measure, "krr.frame");
			setFrameIndex(uint32_t(frame + 1));
			for (auto &pass : mRenderPasses) pass->tick(0.f);
			mScene->update(getFrameIndex(), 0.0);
			renderPasses(measure);
			mNvrhiDevice->runGarbageCollection();
		};
		const json config = mConfig;
		loadConfig(config);
		if (!mNvrhiDevice && !createHeadlessDevice(mDeviceParams))
			throw std::runtime_error("Failed to initialize the offscreen device");
		setFrameIndex(0);
		setSeed(seed);
		initializePasses();
		getRenderContext()->clear();
		if (measure) synchronize();
		const auto setupEnd = Clock::now();
		{
			NvtxRange range(measure, "krr.warmup");
			for (int64_t frame = 0; frame < warmup; ++frame) renderFrame(frame);
			if (measure && warmup) synchronize();
		}
		const auto warmupEnd = Clock::now();
		CaptureScope captureScope(capture, onCaptureBegin, onCaptureEnd);
		captureScope.begin();
		if (mClosed) throw std::runtime_error("Renderer was closed by a capture callback");
		{
			NvtxRange range(measure, "krr.measure");
			const auto renderStart = Clock::now();
			for (int64_t frame = warmup; frame < warmup + frames; ++frame) renderFrame(frame);
			if (measure) synchronize();
			result.timings["render_ms"] = milliseconds(renderStart, Clock::now());
		}
		captureScope.end();
		if (mClosed) throw std::runtime_error("Renderer was closed by a capture callback");
		const auto readbackStart = Clock::now();
		result.image = getRenderContext()->readback();
		const auto readbackEnd = Clock::now();
		for (auto &pass : mRenderPasses) pass->finalize();
		clearScene();
		const auto finalizeEnd = Clock::now();
		result.timings["setup_ms"] = milliseconds(totalStart, setupEnd);
		result.timings["warmup_ms"] = milliseconds(setupEnd, warmupEnd);
		result.timings["readback_ms"] = milliseconds(readbackStart, readbackEnd);
		result.timings["finalize_ms"] = milliseconds(readbackEnd, finalizeEnd);
		result.timings["total_ms"] = milliseconds(totalStart, finalizeEnd);
		return result;
	} catch (...) {
		close();
		throw;
	}
}

RenderApp::RenderApp() = default;
RenderApp::~RenderApp() { close(); }

void RenderApp::backBufferResizing() { 
	if (mpUIRenderer) mpUIRenderer->resizing();
	DeviceManager::backBufferResizing();
}

void Renderer::backBufferResized() {
	DeviceManager::backBufferResized();
	if (mScene)
		mScene->getCamera()->setAspectRatio((float) 
			mDeviceParams.backBufferWidth /
			mDeviceParams.backBufferHeight);
	CUDA_SYNC_CHECK();
}

void RenderApp::backBufferResized() { Renderer::backBufferResized(); }

bool RenderApp::onMouseEvent(io::MouseEvent &mouseEvent) {
	if (io::MouseEvent::Type::Move == mouseEvent.type) {
		sCursorPos[0] = mouseEvent.pos[0] * mDeviceParams.backBufferWidth;
		sCursorPos[1] = mouseEvent.pos[1] * mDeviceParams.backBufferHeight;
	}
	if (mpUIRenderer->onMouseEvent(mouseEvent)) return true;
	if (DeviceManager::onMouseEvent(mouseEvent)) return true;
	if (!sLockCamera && mScene && mScene->onMouseEvent(mouseEvent)) return true;
	return false;
}

bool RenderApp::onKeyEvent(io::KeyboardEvent &keyEvent) {
	if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed) {
		switch (keyEvent.key) { // top-prior operations captured by application
			case io::KeyboardEvent::Key::F1:
				sShowUI = !sShowUI;
				return true;
			case io::KeyboardEvent::Key::F2:
				captureFrame();
				return true;
			case io::KeyboardEvent::Key::Escape:
				gpContext->requestExit();
				return true;
		}
	}
	if (mpUIRenderer->onKeyEvent(keyEvent)) return true;
	if (DeviceManager::onKeyEvent(keyEvent)) return true;
	if (mScene && mScene->onKeyEvent(keyEvent)) return true;
	return false;
}

void Renderer::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	if (mRenderContext) mRenderContext->setScene(scene);
	if (scene)
		scene->getCamera()->setAspectRatio(float(mDeviceParams.backBufferWidth) /
			mDeviceParams.backBufferHeight);
	for (auto p : mRenderPasses) if (p) p->setScene(scene);
}

void RenderApp::run() {
	initialize();
	DeviceManager::runMessageLoop();
	finalize();
}

// Called before beginFrame()...
void RenderApp::tick(double elapsedTime) {
	for (auto it : mRenderPasses) it->tick(float(elapsedTime));
	mpUIRenderer->tick(float(elapsedTime));
	if (mScene) mScene->update(getFrameIndex(), elapsedTime);
}

void RenderApp::render() {
	if (sSaveFrames && getFrameIndex() % sSaveFrameInterval == 0)
		sRequestScreenshot = true;
	DeviceManager::beginFrame();
	
	mpUIRenderer->beginFrame(getRenderContext());
	renderPasses();

	if (sRequestScreenshot) {
		captureFrame(sSaveHDR);
		sRequestScreenshot = false;
	}

	// UI render. This is better done after taking screenshot.
	renderUI();
	mpUIRenderer->render(getRenderContext());
	mpUIRenderer->endFrame(getRenderContext());
	mNvrhiDevice->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, mPresentSemaphore, 0);
	// Blit render buffer, from the render texture (usually HDR) to swapchain texture.
	mCommandList->open();
	mHelperPass->BlitTexture(
		mCommandList, mSwapChainFramebuffers[getCurrentBackBufferIndex()],
							 getRenderContext()->getColorTexture()->getVulkanTexture(),
							 mBindingCache.get());
	mCommandList->close();
	mNvrhiDevice->executeCommandList(mCommandList,
									  nvrhi::CommandQueue::Graphics);

	// If profiler::endframe is called, it queries the gpu time and thus 
	// may cause a cpu-gpu synchronization. Disable it if not necessary.
	if (Profiler::instance().isEnabled()) Profiler::instance().endFrame();
}

void RenderApp::renderUI() {
	static bool showProfiler{};
	static bool showFps{ true };
	static bool showDashboard{ true };
	static bool showCursorPos{ false };
	Profiler::instance().setEnabled(showProfiler);
	if (!sShowUI) return;
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8); // this sets the global transparency of UI windows.
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5);	// this sets the transparency of the main menubar.
	if (ui::BeginMainMenuBar()) {
		ui::PopStyleVar(1);
		if (ui::BeginMenu("Views")) {
			ui::MenuItem("Global UI", NULL, &sShowUI);
			ui::MenuItem("Dashboard", NULL, &showDashboard);
			ui::MenuItem("FPS Counter", NULL, &showFps);
			ui::MenuItem("Show Cursor", NULL, &showCursorPos);
			ui::MenuItem("Profiler", NULL, &showProfiler);
			ui::EndMenu();
		}
		if (ui::BeginMenu("Render")) {
			ui::EndMenu();
		}
		if (ui::BeginMenu("Tools")) {
			ui::MenuItem("Lock Camera", NULL, &sLockCamera);
			if (ui::MenuItem("Save config")) saveConfig("");
			ui::MenuItem("Save HDR", NULL, &sSaveHDR);
			if (ui::MenuItem("Screen shot")) sRequestScreenshot = true;
			ui::EndMenu();
		}
		if (showCursorPos)
			ui::BeginMenu(formatString("[%d, %d]", sCursorPos[0], sCursorPos[1]).c_str(), false);
		if (showFps) 
			ui::BeginMenu(formatString("FPS: %.0lf", 
				1.0 / getAverageFrameTimeSeconds()).c_str(), false);
		ui::EndMainMenuBar();
	}

	if (showDashboard) {
		ui::Begin(KRR_PROJECT_NAME, &showDashboard);
		ui::Checkbox("Profiler", &showProfiler);
		ui::Checkbox("Save HDR", &sSaveHDR);
		ui::SameLine();
		if (ui::Button("Screen shot")) sRequestScreenshot = true;
		if (ui::CollapsingHeader("Configuration")) {
			static char loadConfigBuf[512];
			static char saveConfigBuf[512] = "common/configs/saved_config.json";
			strcpy(loadConfigBuf, mConfigPath.c_str());
			ui::InputText("Load path: ", loadConfigBuf, sizeof(loadConfigBuf));
			if (ui::Button("Load config")) {
				loadConfigFrom(fs::path(loadConfigBuf));
				initializePasses();
			}
			ui::InputText("Save path: ", saveConfigBuf, sizeof(saveConfigBuf));
			if (ui::Button("Save config")) saveConfig(saveConfigBuf);
		}
		if (mScene && ui::CollapsingHeader("Scene")) {
			mScene->renderUI();
		}
		size_t pid = 0;
		for (auto& p : mRenderPasses) {
			ui::PushID(pid++);
			if (p && ui::CollapsingHeader(p->getName().c_str())) 
				p->renderUI();
			ui::PopID();
		}
		ui::End();
	}

	if (Profiler::instance().isEnabled()) {
		if (!mProfilerUI)
			mProfilerUI = ProfilerUI::create(Profiler::instancePtr());
		ui::Begin("Profiler", &showProfiler);
		mProfilerUI->render();
		ui::End();
	}
	ui::PopStyleVar();
}

void RenderApp::captureFrame(bool hdr, fs::path filename) {
	string extension = hdr ? ".exr" : ".png";

	vkrhi::TextureHandle renderTexture = getRenderContext()->getColorTexture()->getVulkanTexture();
	vkrhi::TextureDesc textureDesc	   = renderTexture->getDesc();
	textureDesc.format				   = vkrhi::Format::RGBA32_FLOAT;
	textureDesc.initialState		   = nvrhi::ResourceStates::RenderTarget;
	textureDesc.isRenderTarget		   = true;
	textureDesc.keepInitialState	   = true;
	auto stagingTexture				   = getDevice()->createStagingTexture(
		   textureDesc, vkrhi::CpuAccessMode::Read);
	auto commandList = getDevice()->createCommandList();
	commandList->open();
	commandList->copyTexture(stagingTexture, vkrhi::TextureSlice(),
							 renderTexture, vkrhi::TextureSlice());
	commandList->close();
	getDevice()->executeCommandList(commandList);
	
	size_t pitch;
	auto *data =
		getDevice()->mapStagingTexture(stagingTexture, vkrhi::TextureSlice(),
									   vkrhi::CpuAccessMode::Read, &pitch);

	Image screenshot(getFrameSize(), Image::Format::RGBAfloat);
	memcpy(screenshot.data(), data, screenshot.getSizeInBytes());
	getDevice()->unmapStagingTexture(stagingTexture);

	fs::path filepath(filename);
	if (filename.empty()) // use default path for screen shots
		filepath = File::outputDir() /
				   ("frame_" + std::to_string(getFrameIndex()) + extension);
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	screenshot.saveImage(filepath);
	Log(Success, "Rendering saved to " + filepath.string());
}

void RenderApp::saveConfig(string path) {
	fs::path dirpath = File::resolve("common/configs");
	if (!fs::exists(dirpath))
		fs::create_directories(dirpath);
	fs::path filepath =
		path.empty() ? dirpath / ("config_" + Log::nowToString("%H_%M_%S") + ".json") : path;

	Vector2i resolution;
	getFrameSize(resolution[0], resolution[1]);
	json config			 = mConfig;
	config["resolution"] = resolution;
	config["scene"]		 = *mScene;
	json passes			 = {};
	for (RenderPass::SharedPtr p : mRenderPasses) {
		json p_cfg{ { "name", p->getName() }, { "enable", p->enabled() } };
		passes.push_back(p_cfg);
	}
	config["passes"] = passes;
	File::saveJSON(filepath, config);
	logSuccess("Saved config file to " + filepath.string());
}

void Renderer::loadConfig(const json &config) {
	if (mClosed) throw std::runtime_error("Renderer is closed");
	validateConfig(config);
	Context::ensureInitialized();
	clearScene();
	gpContext->resetState();
	// set global configurations if eligiable
	if (config.contains("global"))
		gpContext->updateGlobalConfig(config.at("global"));

	if (config.contains("output"))
		File::setOutputDir(File::resolve(config.at("output")));

	if (config.contains("renderer")) {
		const json render_config = config.at("renderer");
		sSaveHDR				 = render_config.value("save_hdr", true);
		sLockCamera				 = render_config.value("lock_camera", false);
		sSaveFrames				 = render_config.value("save_frames", false);
		sSaveFrameInterval		 = render_config.value("save_frame_interval", 5);
	}

	if (config.contains("passes")) {
		for (const json &p : config["passes"]) {
			string name = p.at("name");
			Log(Info, "Creating specified render pass: %s", name.c_str());
			RenderPass::SharedPtr pass{};
			if (p.contains("params")) {
				pass = RenderPassFactory::deserizeInstance(name, p.value<json>("params", {}));
			} else {
				pass = RenderPassFactory::createInstance(name);
			}
			if (!pass) throw std::invalid_argument("Cannot configure render pass: " + name);
			pass->setEnable(p.value("enable", true));
			addRenderPassToBack(pass);
		}
	} else
		logWarning("No specified render pass in configuration!");
	Scene::SharedPtr scene { mScene };
	if (config.contains("model")) {
		if (!scene) scene = std::make_shared<Scene>();
		string model = config["model"].get<string>();
		if (!SceneImporter::loadModel(model, scene))
			throw std::runtime_error("Failed to load model: " + model);
	}
	if (config.contains("environment")) {
		if (!scene) Log(Fatal, "Import a model before doing scene configurations!");
		string env	 = config["environment"].get<string>();
		auto texture = Texture::createFromFile(env);
		auto root	 = scene->getSceneGraph()->getRoot();
		auto light	 = std::make_shared<InfiniteLight>(texture);
		scene->getSceneGraph()->attachLeaf(root, light);
	}
	if (config.contains("scene")) {
		if(!scene) scene = std::make_shared<Scene>();
		SceneImporter importer;
		if (!importer.import(config["scene"], scene))
			throw std::runtime_error("Failed to import scene");
	}
	if (scene) mScene = scene;
	if (config.contains("resolution")) {
		Vector2i windowDimension = config.at("resolution");
		mDeviceParams.backBufferWidth = windowDimension[0];
		mDeviceParams.backBufferHeight = windowDimension[1];
		if (mWindow) updateWindowSize();
	}	
	mConfig		= config;
}

void Renderer::loadConfigFrom(fs::path path) {
	json config = File::loadJSON(path);
	loadConfig(config);	
	mConfigPath = path.string();
}

void RenderApp::initialize() { 
	if (!createWindowDeviceAndSwapChain(mDeviceParams, KRR_PROJECT_NAME))
		throw std::runtime_error("Failed to initialize the window device");
	mpUIRenderer = std::make_shared<UIRenderer>(this);
	mpUIRenderer->initialize();
	initializePasses();
}

void RenderApp::finalize() { 
	for (auto pass : mRenderPasses) 
		pass->finalize();
	close();
}

void RenderApp::close() noexcept {
	mpUIRenderer.reset();
	mProfilerUI.reset();
	Renderer::close();
}

NAMESPACE_END(krr)
