#include "renderer.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

void RenderApp::BackBufferResized() {
	DeviceManager::BackBufferResized();
	if (mpScene)
		mpScene->getCamera().setAspectRatio((float) 
			m_DeviceParams.backBufferWidth /
			m_DeviceParams.backBufferHeight);
	CUDA_SYNC_CHECK();
}

bool RenderApp::onMouseEvent(io::MouseEvent &mouseEvent) {
	if (mPaused) return true;
	if (DeviceManager::onMouseEvent(mouseEvent)) return true;
	if (mpScene && mpScene->onMouseEvent(mouseEvent)) return true;
	return false;
}

bool RenderApp::onKeyEvent(io::KeyboardEvent &keyEvent) {
	if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed) {
		switch (keyEvent.key) { // top-prior operations captured by application
			case io::KeyboardEvent::Key::F1:
				mShowUI = !mShowUI;
				return true;
			case io::KeyboardEvent::Key::F3:
				captureFrame();
				return true;
		}
	}
	if (mPaused) return true;
	if (DeviceManager::onKeyEvent(keyEvent)) return true;
	if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;
	return false;
}

void RenderApp::setScene(Scene::SharedPtr scene) {
	mpScene = scene;
	for (auto p : m_RenderPasses)
		if (p) p->setScene(scene);
}

void RenderApp::run() {
	initialize();
	DeviceManager::RunMessageLoop();
	finalize();
}

void RenderApp::renderUI() {
	static bool showProfiler{};
	static bool showFps{ true };
	static bool showDashboard{ true };
	if (!mShowUI)
		return;
	Profiler::instance().setEnabled(showProfiler);
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8); // this sets the global transparency of UI windows.
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5);
	if (ui::BeginMainMenuBar()) {
		ui::PopStyleVar(1);
		if (ui::BeginMenu("Views")) {
			ui::MenuItem("Global UI", NULL, &mShowUI);
			ui::MenuItem("Dashboard", NULL, &showDashboard);
			ui::MenuItem("FPS Counter", NULL, &showFps);
			ui::MenuItem("Profiler", NULL, &showProfiler);
			ui::EndMenu();
		}
		if (ui::BeginMenu("Render")) {
			ui::MenuItem("Pause", NULL, &mPaused);
			ui::EndMenu();
		}
		if (ui::BeginMenu("Tools")) {
			if (ui::MenuItem("Save config"))
				saveConfig("");
			ui::MenuItem("Save HDR", NULL, &mSaveHDR);
			if (ui::MenuItem("Screen shot"))
				captureFrame();
			ui::EndMenu();
		}
		ui::EndMainMenuBar();
	}

	if (showDashboard) {
		ui::Begin(KRR_PROJECT_NAME, &showDashboard);
		ui::Checkbox("Pause", &mPaused);
		ui::SameLine();
		ui::Checkbox("Profiler", &showProfiler);
		ui::Checkbox("Save HDR", &mSaveHDR);
		ui::SameLine();
		if (ui::Button("Screen shot"))
			captureFrame(mSaveHDR);
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
		if (mpScene && ui::CollapsingHeader("Scene"))
			mpScene->renderUI();
		for (auto p : m_RenderPasses)
			if (p && ui::CollapsingHeader(p->getName().c_str()))
				p->renderUI();
		ui::End();
	}

	if (showFps) {
		ImGuiWindowFlags fpsCounterFlags =
			ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
			ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
			ImGuiWindowFlags_NoNav;
		mFrameRate.newFrame();
		ui::SetNextWindowBgAlpha(0.8);
		ui::Begin("FPS Counter", &showFps, fpsCounterFlags);
		ui::Text("FPS: %.1lf", 1000 / mFrameRate.getAverageFrameTime());
		ui::End();
	}

	if (Profiler::instance().isEnabled()) {
		if (!mpProfilerUI)
			mpProfilerUI = ProfilerUI::create(Profiler::instancePtr());
		ui::Begin("Profiler", &showProfiler);
		mpProfilerUI->render();
		ui::End();
	}
	ui::PopStyleVar();
}

void RenderApp::captureFrame(bool hdr, fs::path filename) {
	string extension = hdr ? ".exr" : ".png";

	Log(Error, "Not Implemented!");
}

void RenderApp::saveConfig(string path) {
	fs::path dirpath = File::resolve("common/configs");
	if (!fs::exists(dirpath))
		fs::create_directories(dirpath);
	fs::path filepath =
		path.empty() ? dirpath / ("config_" + Log::nowToString("%H_%M_%S") + ".json") : path;

	json config			 = mConfig;
	config["resolution"] = 0;
	config["scene"]		 = *mpScene;
	json passes			 = {};
	for (RenderPass::SharedPtr p : m_RenderPasses) {
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
		mSpp					 = render_config.value("spp", 0);
		mSaveHDR				 = render_config.value("save_hdr", true);
		mSaveFrames				 = render_config.value("save_frames", false);
		mSaveFrameInterval		 = render_config.value("save_frame_interval", 5);
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
	Scene::SharedPtr scene{ mpScene };
	if (config.contains("model")) {
		scene		 = std::make_shared<Scene>();
		string model = config["model"].get<string>();
		importer::loadScene(model, scene);
	}
	if (config.contains("environment")) {
		if (!scene)
			Log(Fatal, "Import a model before doing scene configurations!");
		string env = config["environment"].get<string>();
		scene->addInfiniteLight(InfiniteLight(env));
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
		m_DeviceParams.backBufferWidth = windowDimension[0];
		m_DeviceParams.backBufferHeight = windowDimension[1];
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
	CreateWindowDeviceAndSwapChain(m_DeviceParams, KRR_PROJECT_NAME);
}

void RenderApp::finalize() { 
	for (auto pass : m_RenderPasses) {
		pass->finalize();
	}
	Shutdown();
}

KRR_NAMESPACE_END