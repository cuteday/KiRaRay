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
	RenderApp(const char title[], Vector2i size) : WindowApp(title, size) { }
	RenderApp(const char title[], Vector2i size, std::vector<RenderPass::SharedPtr> passes)
		:WindowApp(title, size), mpPasses(passes) { }
	
	void resize(const Vector2i& size) override {
		WindowApp::resize(size);
		for (auto p : mpPasses)
			p->resize(size);
		if (mpScene) mpScene->getCamera().setAspectRatio((float)size[0] / size[1]);
		CUDA_SYNC_CHECK();
	}

	// Process signals passed down from direct imgui callback (imgui do not capture it)
	virtual void onMouseEvent(io::MouseEvent& mouseEvent) override {
		if (mPaused) return;
		if (mpScene && mpScene->onMouseEvent(mouseEvent)) return;
		for (auto p : mpPasses)
			if(p->onMouseEvent(mouseEvent)) return;
	}
	
	virtual void onKeyEvent(io::KeyboardEvent &keyEvent) override {
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

	void setScene(Scene::SharedPtr scene) {
		mpScene = scene;
		for (auto p : mpPasses)
			if(p) p->setScene(scene);
	}

	void render() override {
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

	void run() override {
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
				ImGui::Render();
				ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			}

			glfwSwapBuffers(handle);
			glfwPollEvents();
		
			mFrameCount++;
			if (mSpp && mFrameCount >= mSpp) {
				fs::path resultPath = fs::path(mConfigPath).replace_extension("exr");
				captureFrame(false, resultPath);
				break;
			}
		}
	}

	void renderUI() override {
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
					saveConfig();
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
				if (ui::InputInt2("Frame size", (int *) &fbSize))
					resize(fbSize);
				if (ui::Button("Load config"))
					loadConfig("");
				if (ui::Button("Save config"))
					saveConfig();
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

	void draw() override {
		PROFILE("Blit framebuffer");
		WindowApp::draw();
	}

	void captureFrame(bool hdr = false, fs::path filename = "") {
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
		logSuccess("Saved screenshot to " + filepath.string());
	}

	void saveConfig() {
		fs::path dirpath = File::resolve("common/configs"); 
		if (!fs::exists(dirpath))
			fs::create_directories(dirpath);
		fs::path filepath = dirpath / ("config_" + Log::nowToString("%H_%M_%S") + ".json");
		std::ofstream ofs(filepath);
		json config = mConfig;
		config["resolution"] = fbSize;
		config["scene"]		 = *mpScene;
		ofs << config;
		ofs.close();
		logSuccess("Saved config file to " + filepath.string());
	}

	void loadConfig(fs::path path) {
		if (!fs::exists(path)) {
			logError("Config file not found at " + path.string());
			return;
		}
		std::ifstream f(path);
		json config = json::parse(f, nullptr, true);
		mSpp		= config["spp"].get<int>();
		if (config.contains("model")) {
			string model		   = config["model"].get<string>();
			Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
			AssimpImporter().import(model, scene);
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
		resize(config["resolution"].get<Vector2i>());
		mConfig = config;
		mConfigPath = path.string();
	}

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