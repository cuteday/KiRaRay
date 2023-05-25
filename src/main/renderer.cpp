#include "renderer.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

namespace { // status bits
static bool sShowUI				 = true;
static bool sSaveHDR			 = false;
static bool sSaveFrames			 = false;
static bool sRequestScreenshot	 = false;
static size_t sSaveFrameInterval = 2;
}

RenderApp::RenderApp() {
	if (!gpContext) gpContext = std::make_shared<Context>();
}

void RenderApp::BackBufferResizing() { 
	if (mpUIRenderer) mpUIRenderer->resizing();
	DeviceManager::BackBufferResizing();
}

void RenderApp::BackBufferResized() {
	DeviceManager::BackBufferResized();
	if (mScene)
		mScene->getCamera().setAspectRatio((float) 
			mDeviceParams.backBufferWidth /
			mDeviceParams.backBufferHeight);
	CUDA_SYNC_CHECK();
}

bool RenderApp::onMouseEvent(io::MouseEvent &mouseEvent) {
	if (mpUIRenderer->onMouseEvent(mouseEvent)) return true;
	if (DeviceManager::onMouseEvent(mouseEvent)) return true;
	if (mScene && mScene->onMouseEvent(mouseEvent)) return true;
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
		}
	}
	if (mpUIRenderer->onKeyEvent(keyEvent)) return true;
	if (DeviceManager::onKeyEvent(keyEvent)) return true;
	if (mScene && mScene->onKeyEvent(keyEvent)) return true;
	return false;
}

void RenderApp::setScene(Scene::SharedPtr scene) {
	mScene = scene;
	for (auto p : mRenderPasses) if (p) p->setScene(scene);
}

void RenderApp::run() {
	initialize();
	DeviceManager::RunMessageLoop();
	finalize();
}

void RenderApp::Tick(double elapsedTime) {
	for (auto it : mRenderPasses) it->tick(float(elapsedTime));
	mpUIRenderer->tick(float(elapsedTime));
}

void RenderApp::Render() {
	if (sSaveFrames && GetFrameIndex() % sSaveFrameInterval == 0)
		sRequestScreenshot = true;

	if (mScene) mScene->update(GetFrameIndex());
	BeginFrame();
	auto framebuffer = mRenderFramebuffers[GetCurrentBackBufferIndex()];
	mpUIRenderer->beginFrame();
	for (auto it : mRenderPasses) it->beginFrame();
	for (auto it : mRenderPasses) {
		if (it->isCudaPass()) framebuffer->vulkanUpdateCuda(mGraphicsSemaphore);
		it->render(framebuffer);
		if (it->isCudaPass()) framebuffer->cudaUpdateVulkan();
	}
	for (auto it : mRenderPasses) it->endFrame();

	if (sRequestScreenshot) {
		captureFrame(sSaveHDR);
		sRequestScreenshot = false;
	}

	// UI render. This is better done after taking screenshot.
	renderUI();
	mpUIRenderer->render(framebuffer);
	mpUIRenderer->endFrame();
	mNvrhiDevice->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, mPresentSemaphore, 0);
	// Blit render buffer, from the render texture (usually HDR) to swapchain texture.
	mCommandList->open();
	mHelperPass->BlitTexture(
		mCommandList, mSwapChainFramebuffers[GetCurrentBackBufferIndex()],
		GetCurrentRenderImage(), mBindingCache.get());
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
	if (!sShowUI)
		return;
	Profiler::instance().setEnabled(showProfiler);
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8); // this sets the global transparency of UI windows.
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5);	// this sets the transparency of the main menubar.
	if (ui::BeginMainMenuBar()) {
		ui::PopStyleVar(1);
		if (ui::BeginMenu("Views")) {
			ui::MenuItem("Global UI", NULL, &sShowUI);
			ui::MenuItem("Dashboard", NULL, &showDashboard);
			ui::MenuItem("FPS Counter", NULL, &showFps);
			ui::MenuItem("Profiler", NULL, &showProfiler);
			ui::EndMenu();
		}
		if (ui::BeginMenu("Render")) {
			ui::EndMenu();
		}
		if (ui::BeginMenu("Tools")) {
			if (ui::MenuItem("Save config")) saveConfig("");
			ui::MenuItem("Save HDR", NULL, &sSaveHDR);
			if (ui::MenuItem("Screen shot")) sRequestScreenshot = true;
			ui::EndMenu();
		}
		if (showFps) 
			ui::BeginMenu(formatString("FPS: %.0lf", 
				1.0 / GetAverageFrameTimeSeconds()).c_str(), false);
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
			if (ui::Button("Load config"))
				loadConfig(fs::path(loadConfigBuf));
			ui::InputText("Save path: ", saveConfigBuf, sizeof(saveConfigBuf));
			if (ui::Button("Save config"))
				saveConfig(saveConfigBuf);
		}
		if (mScene && ui::CollapsingHeader("Scene"))
			mScene->renderUI();
		for (auto p : mRenderPasses)
			if (p && ui::CollapsingHeader(p->getName().c_str()))
				p->renderUI();
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

	vkrhi::TextureHandle renderTexture = GetCurrentRenderImage();
	vkrhi::TextureDesc textureDesc	   = renderTexture->getDesc();
	textureDesc.format				   = vkrhi::Format::RGBA32_FLOAT;
	textureDesc.initialState		   = nvrhi::ResourceStates::RenderTarget;
	textureDesc.isRenderTarget		   = true;
	textureDesc.keepInitialState	   = true;
	auto stagingTexture				   = GetDevice()->createStagingTexture(
		   textureDesc, vkrhi::CpuAccessMode::Read);
	auto commandList = GetDevice()->createCommandList();
	commandList->open();
	commandList->copyTexture(stagingTexture, vkrhi::TextureSlice(),
							 renderTexture, vkrhi::TextureSlice());
	commandList->close();
	GetDevice()->executeCommandList(commandList);
	
	size_t pitch;
	auto *data =
		GetDevice()->mapStagingTexture(stagingTexture, vkrhi::TextureSlice(),
									   vkrhi::CpuAccessMode::Read, &pitch);

	Image screenshot(GetWindowDimensions(), Image::Format::RGBAfloat);
	memcpy(screenshot.data(), data, screenshot.getSizeInBytes());
	GetDevice()->unmapStagingTexture(stagingTexture);

	fs::path filepath(filename);
	if (filename.empty()) // use default path for screen shots
		filepath = File::outputDir() /
				   ("frame_" + std::to_string(GetFrameIndex()) + extension);
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	screenshot.saveImage(filepath);
	logSuccess("Rendering saved to " + filepath.string());
}

void RenderApp::saveConfig(string path) {
	fs::path dirpath = File::resolve("common/configs");
	if (!fs::exists(dirpath))
		fs::create_directories(dirpath);
	fs::path filepath =
		path.empty() ? dirpath / ("config_" + Log::nowToString("%H_%M_%S") + ".json") : path;

	Vector2i resolution;
	GetWindowDimensions(resolution[0], resolution[1]);
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

void RenderApp::loadConfig(const json config) {
	// set global configurations if eligiable
	if (config.contains("global"))
		gpContext->updateGlobalConfig(config.at("global"));

	if (config.contains("output"))
		File::setOutputDir(File::resolve(config.at("output")));

	if (config.contains("renderer")) {
		const json render_config = config.at("renderer");
		sSaveHDR				 = render_config.value("save_hdr", true);
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
			pass->setEnable(p.value("enable", true));
			AddRenderPassToBack(pass);
		}
	} else
		logWarning("No specified render pass in configuration!");
	Scene::SharedPtr scene { mScene };
	if (config.contains("model")) {
		scene		 = std::make_shared<Scene>();
		string model = config["model"].get<string>();
		importer::loadScene(model, scene);
	}
	if (config.contains("environment")) {
		if (!scene)
			Log(Fatal, "Import a model before doing scene configurations!");
		string env = config["environment"].get<string>();
		scene->addEnvironmentMap(Texture::createFromFile(env));
	}
	if (config.contains("scene")) {
		if (!scene)
			Log(Fatal, "Import a model before doing scene configurations!");
		scene->loadConfig(config["scene"]);
	}
	if (scene)
		setScene(scene);
	if (config.contains("resolution")) {
		Vector2i windowDimension = config.at("resolution");
		mDeviceParams.backBufferWidth = windowDimension[0];
		mDeviceParams.backBufferHeight = windowDimension[1];
		UpdateWindowSize();
	}	
	mConfig		= config;
}

void RenderApp::loadConfigFrom(fs::path path) {
	json config = File::loadJSON(path);
	loadConfig(config);	
	mConfigPath = path.string();
}

void RenderApp::initialize() { 
	CreateWindowDeviceAndSwapChain(mDeviceParams, KRR_PROJECT_NAME);
	mpUIRenderer = std::make_shared<UIRenderer>(this);
	mpUIRenderer->initialize();
	for (auto pass : mRenderPasses) pass->initialize();
	mNvrhiDevice->waitForIdle();
}

void RenderApp::finalize() { 
	for (auto pass : mRenderPasses) 
		pass->finalize();
	mpUIRenderer.reset();
	mRenderPasses.clear();
	mScene.reset();
	// Destroy created vulkan resources before destroy vulkan device
	Shutdown();
}

KRR_NAMESPACE_END