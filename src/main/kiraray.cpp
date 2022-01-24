#include <filesystem>

#include "kiraray.h"

#include "common.h"
#include "GLFW/glfw3.h"

#include "renderer.h"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char* argv[]) {

	logSuccess("Kiraray::Main Hello, world!");
	logInfo("Kiraray::Main Working directory: " + std::filesystem::current_path().string());

	RenderApp app("Kiraray", { 1920, 1080 });
	
	Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
	scene->createFromFile("../../../../../common/assets/cbox/CornellBox-Original.obj");
	
	app.renderer().setScene(scene);
	app.run();

	return 0;
}

KRR_NAMESPACE_END