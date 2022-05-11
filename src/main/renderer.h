#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/buffer.h"
#include "device/context.h"
#include "render/postprocess.h"

KRR_NAMESPACE_BEGIN

class RenderApp : public WindowApp{
public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) { }
	RenderApp(const char title[], vec2i size, std::vector<RenderPass::SharedPtr> passes)
		:WindowApp(title, size), mpPasses(passes) { }

	void resize(const vec2i& size) override {
		if (!mpScene) return;
		mRenderBuffer.resize(size.x * size.y * sizeof(vec4f));
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
		mpScene->update();	// maybe this should be put into beginFrame() or sth...
		for (auto p : mpPasses)
			if (p) { 
				p->beginFrame(mRenderBuffer);
				p->render(mRenderBuffer); 
			}
	}

	void renderUI() override{
		static bool saveHdr = false;
		ui::Begin(KRR_PROJECT_NAME);
		if (ui::Button("Screen shot"))
			captureFrame(saveHdr);
		ui::SameLine();
		ui::Checkbox("Save HDR", &saveHdr);
		//ui::SetTooltip("Disable tone mapping (render to linear color space) if you want to save HDR");
		if(ui::InputInt2("Frame size", (int*)&fbSize))
			resize(fbSize);
		if (mpScene) mpScene->renderUI();
		for (auto p : mpPasses)
			if (p) p->renderUI();
		ui::End();
	}

	void draw() override {
		mRenderBuffer.copy_to_device(fbPointer, fbSize.x * fbSize.y);
		WindowApp::draw();
	}

private:
	void captureFrame(bool hdr = false) {
		string extension = hdr ? ".exr" : ".png";
		Image image(fbSize, Image::Format::RGBAfloat);
		mRenderBuffer.copy_to_host(image.data(), fbSize.x * fbSize.y * 4 * sizeof(float));
		fs::path filepath = File::resolve("common/images") / ("krr_" + Log::nowToString("%H_%M_%S") + extension);
		image.saveImage(filepath.string());
		logSuccess("Saved screen shot to " + filepath.string());
	}

	std::vector<RenderPass::SharedPtr> mpPasses;
	Scene::SharedPtr mpScene;
	CUDABuffer mRenderBuffer;
};

KRR_NAMESPACE_END