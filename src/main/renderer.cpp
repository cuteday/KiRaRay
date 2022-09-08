#include "common.h"
#include "renderer.h"

KRR_NAMESPACE_BEGIN
	
void RenderApp::resize(const Vector2i size) {
	WindowApp::resize(size);
	for (auto p : mpPasses)
		p->resize(size);
	if (mpScene) mpScene->getCamera().setAspectRatio((float)size[0] / size[1]);
	CUDA_SYNC_CHECK();
}

// Process signals passed down from direct imgui callback (imgui do not capture it)
void RenderApp::onMouseEvent(io::MouseEvent& mouseEvent) {
	if (mPaused) return;
	if (mpScene && mpScene->onMouseEvent(mouseEvent)) return;
	for (auto p : mpPasses)
		if(p->onMouseEvent(mouseEvent)) return;
}

void RenderApp::onKeyEvent(io::KeyboardEvent &keyEvent) {
	if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed) {
		switch (keyEvent.key) {		// top-prior operations captured by application
		case io::KeyboardEvent::Key::F1:
			mShowUI = !mShowUI;
			return;
		case io::KeyboardEvent::Key::F3:
			captureFrame();
			return;
		}
	}
	if (mPaused) return;
	// passing down signals...
	if (mpScene && mpScene->onKeyEvent(keyEvent)) return;	
	for (auto p : mpPasses)
		if(p->onKeyEvent(keyEvent)) return;
}

void RenderApp::setScene(Scene::SharedPtr scene) {
	mpScene = scene;
	for (auto p : mpPasses)
		if(p) p->setScene(scene);
}

void RenderApp::render()  {
	if (!mpScene) return;
	if (!mPaused) {		// Froze all updates if paused
		mpScene->update();
		for (auto p : mpPasses)
			if (p) {
				p->beginFrame(fbBuffer);
				p->render(fbBuffer);
				p->endFrame(fbBuffer);
			}
	}
	if (Profiler::instance().isEnabled()) Profiler::instance().endFrame();
}

void RenderApp::run() {
	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(Vector2i(width, height));

	while (!glfwWindowShouldClose(handle)) {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render();

		draw();
		renderUI();
		{
			PROFILE("Draw UI");
			ui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ui::GetDrawData());
			if (ui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
				ui::UpdatePlatformWindows();
				ui::RenderPlatformWindowsDefault();
			}
		}

		glfwSwapBuffers(handle);
		glfwPollEvents();
	
		mFrameCount++;
		if (mSpp && mFrameCount >= mSpp) {
			Log(Info, "Render process finished, saving results and quitting...");
			fs::path resultPath = fs::path(mConfigPath).replace_extension("exr");
			captureFrame(false, resultPath);
			break;
		}
	}
}

void RenderApp::renderUI() {
	static bool saveHdr{};
	static bool showProfiler{};
	static bool showFps{ true };
	static bool showDashboard{ true };
	if (!mShowUI) return;
	Profiler::instance().setEnabled(showProfiler);
	ui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8);		// this sets the global transparency of UI windows.
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
			ui::MenuItem("Save HDR", NULL, &saveHdr);
			if (ui::MenuItem("Screen shot")) 
				captureFrame();
			ui::EndMenu();
		}
		ui::EndMainMenuBar();
	}

	if(showDashboard){
		ui::Begin(KRR_PROJECT_NAME, &showDashboard);
		ui::Checkbox("Pause", &mPaused); ui::SameLine();
		ui::Checkbox("Profiler", &showProfiler);
		ui::Checkbox("Save HDR", &saveHdr); ui::SameLine();
		if (ui::Button("Screen shot"))
			captureFrame(saveHdr);
		if (ui::CollapsingHeader("Configuration")) {
			static char loadConfigBuf[512];
			static char saveConfigBuf[512] = "common/configs/saved_config.json"; 
			strcpy(loadConfigBuf, mConfigPath.c_str());
			if (ui::InputInt2("Frame size", (int *) &fbSize))
				resize(fbSize);
			ui::InputText("Load path: ", loadConfigBuf, 1024); 
			if (ui::Button("Load config"))
				loadConfig(loadConfigBuf);
			ui::InputText("Save path: ", saveConfigBuf, 1024);
			if (ui::Button("Save config"))
				saveConfig(saveConfigBuf);
		}
		//if (ui::CollapsingHeader("Performance")) {
		//	mFrameRate.plotFrameTimeGraph();
		//}
		if (mpScene && ui::CollapsingHeader("Scene")) 
			mpScene->renderUI();
		for (auto p : mpPasses)
			if (p && ui::CollapsingHeader(p->getName().c_str()) )
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
		if (!mpProfilerUI) mpProfilerUI = ProfilerUI::create(Profiler::instancePtr());
		ui::Begin("Profiler", &showProfiler);
		mpProfilerUI->render();
		ui::End();
	}	
	ui::PopStyleVar();
}

void RenderApp::draw() {
	PROFILE("Blit framebuffer");
	WindowApp::draw();
}

void RenderApp::captureFrame(bool hdr, fs::path filename) {
	string extension = hdr ? ".exr" : ".png";
	Image image(fbSize, Image::Format::RGBAfloat);
	fbBuffer.copy_to_host(image.data(), fbSize[0] * fbSize[1] * 4 * sizeof(float));
	fs::path filepath(filename);
	if (filename.empty())		// use default path for screen shots
		filepath = File::resolve("common/images") / ("screenshot_" + Log::nowToString("%H_%M_%S") + extension);
	fs::path dirpath  = File::resolve("common/images"); 
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	image.saveImage(filepath);
	logSuccess("Rendering saved to " + filepath.string());
}

void RenderApp::saveConfig(string path) {
	fs::path dirpath = File::resolve("common/configs"); 
	if (!fs::exists(dirpath))
		fs::create_directories(dirpath);
	fs::path filepath = path.empty()? dirpath / ("config_" + Log::nowToString("%H_%M_%S") + ".json") : path;
	std::ofstream ofs(filepath);
	json config = mConfig;
	config["resolution"] = fbSize;
	config["scene"]		 = *mpScene;
	json passes			 = {};
	for (RenderPass::SharedPtr p : mpPasses) {
		json p_cfg{ 
			{ "name", p->getName() }, 
			{ "enable", p->enabled() } 
		};
		passes.push_back(p_cfg);
	}
	config["passes"] = passes;
	ofs << config;
	ofs.close();
	logSuccess("Saved config file to " + filepath.string());
}

void RenderApp::loadConfig(fs::path path) {
	if (!fs::exists(path)) {
		logError("Config file not found at " + path.string());
		return;
	}
	std::ifstream f(path);
	json config = json::parse(f, nullptr, true);
	mSpp = config.value("spp", 0);
	if (config.contains("passes")) {
		mpPasses.clear();
		for (const json &p : config["passes"]) {
			string name				   = p.at("name");
			Log(Info, "Creating specified render pass: %s", name.c_str());
			RenderPass::SharedPtr pass = RenderPassFactory::createInstance(name);
			if (p.contains("parameters")) {
				//*pass = p["parameters"];	
			}
			pass->setEnable(p.value("enable", true));
			mpPasses.push_back(pass);
		}
	} else
		logWarning("No specified render pass in configuration!");
	if (config.contains("model")) {
		string model		   = config["model"].get<string>();
		Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
		importer::loadScene(model, scene);
		setScene(scene);
	}
	if (config.contains("environment")) {
		if (!mpScene)
			Log(Fatal, "Import a model before doing scene configurations!");
		string env = config["environment"].get<string>();
		mpScene->addInfiniteLight(InfiniteLight(env));
	}
	if (config.contains("scene")) {
		if (!mpScene)
			Log(Fatal, "Import a model before doing scene configurations!");
		mpScene->loadConfig(config["scene"]);	
	}
	if (config.contains("resolution"))
		resize(config.value("resolution", fbSize));
	mConfig = config;
	mConfigPath = path.string();
}

KRR_NAMESPACE_END