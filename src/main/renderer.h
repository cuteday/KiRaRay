#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/buffer.h"
#include "device/context.h"
#include "render/profiler/ui.h"

KRR_NAMESPACE_BEGIN

class RenderApp : public WindowApp{
public:
	RenderApp(const char title[], vec2i size) : WindowApp(title, size) { }
	RenderApp(const char title[], vec2i size, std::vector<RenderPass::SharedPtr> passes)
		:WindowApp(title, size), mpPasses(passes) { }

	void resize(const vec2i& size) override {
		if (!mpScene) return;
		WindowApp::resize(size);
		for (auto p : mpPasses)
			p->resize(size);
		mpScene->getCamera().setAspectRatio((float)size.x / size.y);
	}

	virtual void onMouseEvent(io::MouseEvent& mouseEvent) override {
		if (mpScene && mpScene->onMouseEvent(mouseEvent)) return;
		for (auto p : mpPasses)
			if(p->onMouseEvent(mouseEvent)) return;
	}
	
	virtual void onKeyEvent(io::KeyboardEvent &keyEvent) override {
		if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed) {
			switch (keyEvent.key) {		// top-prior operations captured by application
			case io::KeyboardEvent::Key::F1:
				captureFrame();
				return;
			case io::KeyboardEvent::Key::F2:
				Profiler::instance().setEnabled(!Profiler::instance().isEnabled());
				return;
			}
		}
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
		mpScene->update();
		for (auto p : mpPasses)
			if (p) { 
				p->beginFrame(fbBuffer);
				p->render(fbBuffer);
			}
		if (Profiler::instance().isEnabled()) Profiler::instance().endFrame();
	}

	void renderUI() override{
		static bool saveHdr = false;
		ui::Begin(KRR_PROJECT_NAME);
		if (ui::Button(Profiler::instance().isEnabled() ?"Hide profiler" : "Show profiler"))
			Profiler::instance().setEnabled(!Profiler::instance().isEnabled());
		PROFILE("Draw UI");
		ui::SameLine();
		if (ui::Button("Screen shot"))
			captureFrame(saveHdr);
		ui::SameLine();
		ui::Checkbox("Save HDR", &saveHdr);
		if(ui::InputInt2("Frame size", (int*)&fbSize))
			resize(fbSize);
		if (mpScene) mpScene->renderUI();
		for (auto p : mpPasses)
			if (p) p->renderUI();
		ui::End();
		if (Profiler::instance().isEnabled()) {
			if (!mpProfilerUI) mpProfilerUI = ProfilerUI::create(Profiler::instancePtr());
			ui::Begin("Profiler");
			mpProfilerUI->render();
			ui::End();
		}	
	}

	void draw() override {
		PROFILE("Blit framebuffer");
		WindowApp::draw();
	}

private:
	void captureFrame(bool hdr = false) {
		string extension = hdr ? ".exr" : ".png";
		Image image(fbSize, Image::Format::RGBAfloat);
		fbBuffer.copy_to_host(image.data(), fbSize.x * fbSize.y * 4 * sizeof(float));
		fs::path filepath = File::resolve("common/images") / ("krr_" + Log::nowToString("%H_%M_%S") + extension);
		image.saveImage(filepath);
		logSuccess("Saved screen shot to " + filepath.string());
	}

	std::vector<RenderPass::SharedPtr> mpPasses;
	Scene::SharedPtr mpScene;
	ProfilerUI::UniquePtr mpProfilerUI;
};

KRR_NAMESPACE_END