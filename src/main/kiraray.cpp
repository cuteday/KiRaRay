#include <filesystem>

#include "file.h"
#include "kiraray.h"
#include "renderer.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

namespace {
	inline void locateWorkingDirectory() {
		std::filesystem::current_path(File::cwd());
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
	const string sceneFile = "common/scenes/rungholt/rungholt.obj";
	//const string sceneFile = "common/scenes/little_witch/03.obj";
	//const string sceneFile = "common/scenes/sponza/sponza.obj";
	//const string sceneFile = "common/scenes/living_room/living_room.obj";
	//const string sceneFile = "common/scenes/breakfast_room/breakfast_room.obj";

	//const string iblFile = "common/assets/Tropical_Beach.hdr";
	const string iblFile = "common/assets/Playa_Sunrise.exr";
	//const string iblFile = "common/assets/Mono_Lake_B.hdr";
	//const string iblFile = "common/assets/Ridgecrest_Road.hdr";

	try {
		BxDF bsdf = new DiffuseBrdf();
		ShadingData sd;
		bsdf.setup(sd);

		gpContext = Context::SharedPtr(new Context());
		RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 });

		Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
		scene->getEnvLight()->setImage(iblFile);
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