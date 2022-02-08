#include <filesystem>

#include "kiraray.h"
#include "renderer.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

namespace {
	inline void locateWorkingDirectory() {
		auto current_directory = std::filesystem::current_path();
		while(current_directory.filename().string().find(KRR_PROJECT_NAME) == string::npos) {
			if (current_directory.parent_path() == current_directory) break;
			current_directory = current_directory.parent_path();
		}
		if (current_directory.filename().string().find(KRR_PROJECT_NAME) == string::npos)
			logFatal("Kiraray::Can not locate root working directory, "
				"make sure you are with in the project folder.");
		std::filesystem::current_path(current_directory);
		logInfo("Main Working directory: " + std::filesystem::current_path().string());
	}
}

extern "C" int main(int argc, char* argv[]) {

	locateWorkingDirectory();
	logSuccess("Kiraray::Main Hello, world!");
#ifdef KRR_DEBUG_BUILD
	logWarning("Kiraray::Running in debug mode!");
#endif

	//const string sceneFile = "common/scenes/cbox/CornellBox-Original.obj";
	const string sceneFile = "common/scenes/sponza/sponza.obj";
	//const string sceneFile = "common/scenes/living_room/living_room.obj";
	//const string sceneFile = "common/scenes/breakfast_room/breakfast_room.obj";

	try {
		BxDF bsdf = new DiffuseBxDF();
		ShadingData sd;
		bsdf.setup(sd);

		gpContext = Context::SharedPtr(new Context());
		RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 });

		Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
		AssimpImporter importer;
		importer.import(sceneFile, scene);
		app.setScene(scene);
		app.run();
	} 
	catch(std::exception e) {
		logFatal("Kiraray::Unhandled exception: " + string(e.what()));
	}

	return 0;
}

KRR_NAMESPACE_END