#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/buffer.h"
#include "device/context.h"
#include "render/profiler/ui.h"
#include "render/profiler/fps.h"

KRR_NAMESPACE_BEGIN

class RenderApp : public WindowApp{
public:
	RenderApp(const char title[], Vector2i size) : WindowApp(title, size) { }
	RenderApp(const char title[], Vector2i size, std::vector<RenderPass::SharedPtr> passes)
		:WindowApp(title, size), mpPasses(passes) { }

	void resize(const Vector2i& size) override {
		if (!mpScene) return;
		WindowApp::resize(size);
		for (auto p : mpPasses)
			p->resize(size);
		mpScene->getCamera().setAspectRatio((float)size[0] / size[1]);
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
				ui::MenuItem("Save HDR", NULL, &saveHdr);
				if (ui::MenuItem("Screen shot")) captureFrame();
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

			if (ui::InputInt2("Frame size", (int*)&fbSize))
				resize(fbSize);
			
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

private:
	void captureFrame(bool hdr = false) {
		string extension = hdr ? ".exr" : ".png";
		Image image(fbSize, Image::Format::RGBAfloat);
		fbBuffer.copy_to_host(image.data(), fbSize[0] * fbSize[1] * 4 * sizeof(float));
		fs::path filepath = File::resolve("common/images") / ("krr_" + Log::nowToString("%H_%M_%S") + extension);
		image.saveImage(filepath);
		logSuccess("Saved screenshot to " + filepath.string());
	}

	bool mShowUI{ true };
	bool mPaused{ false };
	FrameRate mFrameRate;
	std::vector<RenderPass::SharedPtr> mpPasses;
	Scene::SharedPtr mpScene;
	ProfilerUI::UniquePtr mpProfilerUI;
};

KRR_NAMESPACE_END