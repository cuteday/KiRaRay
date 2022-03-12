#include <filesystem>

#include "file.h"
#include "kiraray.h"
#include "renderer.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char* argv[]) {

	logSuccess("Kiraray::Main Hello, world!");

	std::filesystem::current_path(File::cwd());
	logInfo("Working directory: " + string(KRR_PROJECT_DIR));
	logInfo("Kiraray build type: " + string(KRR_BUILD_TYPE));
#ifdef KRR_DEBUG_BUILD
	logWarning("Kiraray::Running in debug mode!");
#endif
#if KRR_PLATFORM_UNKNOWN
	logError("Kiraray::Running on unsupported platform!");
#endif

	string sceneFile = "common/scenes/cbox/CornellBox-Original.obj";
	//sceneFile = "common/scenes/cbox/CornellBox-Sphere.obj";
	//sceneFile = "common/scenes/rungholt/rungholt.obj";
	//sceneFile = "common/scenes/sponza/sponza.obj";
	//sceneFile = "common/scenes/living_room/living_room.obj";
	//sceneFile = "common/scenes/fireplace_room/fireplace_room.obj";
	//sceneFile = "common/scenes/salle_de_bain/salle_de_bain.obj";
	//sceneFile = "common/scenes/bedroom/iscv2.obj";
	//sceneFile = "common/scenes/breakfast_room/breakfast_room.obj";

	string iblFile = "common/assets/snowwhite.jpg";
	//iblFile = "common/assets/Tropical_Beach.hdr";
	//iblFile = "common/assets/Playa_Sunrise.exr";
	//iblFile = "common/assets/Mono_Lake_B.hdr";
	//iblFile = "common/assets/Ridgecrest_Road.hdr";

#ifndef KRR_DEBUG_BUILD
	try {
#endif
		gpContext = Context::SharedPtr(new Context());
		RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 });

		Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
		scene->getEnvLight()->setImage(iblFile);
		AssimpImporter importer;
		importer.import(sceneFile, scene);
		app.setScene(scene);
		app.run();
#ifndef KRR_DEBUG_BUILD
	} catch (std::exception e) {
		logFatal("Kiraray::Unhandled exception: " + string(e.what()));
	}
#endif // KRR_DEBUG_BUILD

	return EXIT_SUCCESS;
}

KRR_NAMESPACE_END