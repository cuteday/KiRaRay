#pragma once

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "gpu/buffer.h"
#include "gpu/context.h"
#include "render/path.h"
#include "render/postprocess.h"

KRR_NAMESPACE_BEGIN

class PathTracer: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<PathTracer>;

	PathTracer();

	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();
	void buildAS();

	bool onKeyEvent(const KeyboardEvent& keyEvent) override;
	bool onMouseEvent(const MouseEvent& mouseEvent) override;
	void renderUI() override;
	void render(CUDABuffer& frame) override;
	void resize(const vec2i& size) override { 
		mFrameSize = size;
		launchParams.fbSize = size; 
	}

	void setScene(Scene::SharedPtr scene) override {
		mpScene = scene;
		buildAS();
		buildSBT();
		logSuccess("Scene set...");
	}

private:
	OptixPipeline               pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions    pipelineLinkOptions = {};

	OptixModule                 module;
	OptixModuleCompileOptions   moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	LaunchParamsPT launchParams;
	CUDABuffer   launchParamsBuffer;

	CUDABuffer accelBuffer;
};

class RenderApp : public WindowApp{
public:

	RenderApp(const char title[], vec2i size) : WindowApp(title, size) {
		mpPasses = {
			PathTracer::SharedPtr(new PathTracer()),
			AccumulatePass::SharedPtr(new AccumulatePass()),
			ToneMappingPass::SharedPtr(new ToneMappingPass())
		};
	}

	void initialize();

	void resize(const vec2i& size) override {
		mRenderBuffer.resize(size.x * size.y * sizeof(vec4f));
		WindowApp::resize(size);
		for (auto p : mpPasses)
			p->resize(size);
		mpScene->getCamera()->setAspectRatio((float)size.x / size.y);
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
		mpScene->update();	// maybe this should be put into beginFrame() or sth...
		for (auto p : mpPasses)
			if (p) p->render(mRenderBuffer);
	}

	void renderUI() override{
		static bool saveHdr = false;

		ui::Begin(KRR_PROJECT_NAME);
		ui::Text("Window size: %d %d", fbSize.x, fbSize.y);
		if(ui::Button("Screen shot"))
			captureFrame(saveHdr); 
		ui::SameLine();
		ui::Checkbox("Save HDR", &saveHdr);
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
		logInfo("Saved screen shot to " + filepath.string());
	}

	std::vector<RenderPass::SharedPtr> mpPasses;
	Scene::SharedPtr mpScene;
	CUDABuffer mRenderBuffer;
};

KRR_NAMESPACE_END