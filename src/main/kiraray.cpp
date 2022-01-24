#include <filesystem>

#include "kiraray.h"

#include "common.h"
#include "GLFW/glfw3.h"

#include "renderer.h"

KRR_NAMESPACE_BEGIN

namespace {
	inline void locateWorkingDirectory() {
		auto current_directory = std::filesystem::current_path();
		while(current_directory.filename().string().find("KiRaRay") == string::npos) {
			if (current_directory.parent_path() == current_directory) break;
			current_directory = current_directory.parent_path();
			//logInfo(current_directory.string());
		}
		if (current_directory.filename().string().find("KiRaRay") == string::npos)
			logFatal("Kiraray::Can't locate root working directory, "
				"make sure you are with in the project folder.");
		std::filesystem::current_path(current_directory);
		logInfo("Kiraray::Main Working directory: " + std::filesystem::current_path().string());
	}
}

extern "C" int main(int argc, char* argv[]) {

	logSuccess("Kiraray::Main Hello, world!");
	locateWorkingDirectory();

	RenderApp app("Kiraray", { 1920, 1080 });
	
	Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
	scene->createFromFile("common/assets/cbox/CornellBox-Original.obj");
	
	app.renderer().setScene(scene);
	app.run();

	return 0;
}

KRR_NAMESPACE_END