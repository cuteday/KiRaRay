#include <filesystem>

#include "file.h"
#include "kiraray.h"

#include "main/renderer.h"
#include "render/passes/accumulate.h"
#include "render/passes/tonemapping.h"
#include "render/path/pathtracer.h"
#include "render/wavefront/integrator.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char *argv[]) {
	std::filesystem::current_path(File::cwd());
	logInfo("Working directory: " + string(KRR_PROJECT_DIR));
	logInfo("Kiraray build type: " + string(KRR_BUILD_TYPE));
#ifdef KRR_DEBUG_BUILD
	logWarning("Running in debug mode, the performance may be extremely slow.\n"
				"\t\tSwitch to Release build for normal performance!");
#endif
	string sceneFile = "common/assets/scenes/cbox/cbox.obj";
	string iblFile	 = "common/assets/textures/snowwhite.jpg";

	try {
		gpContext = Context::SharedPtr(new Context());
		RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 },
					  { RenderPass::SharedPtr(new PathTracer()),
						//RenderPass::SharedPtr(new WavefrontPathTracer()),
						RenderPass::SharedPtr(new AccumulatePass()),
						RenderPass::SharedPtr(new ToneMappingPass()) });
		Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
		scene->addInfiniteLight(InfiniteLight(iblFile));
		AssimpImporter importer;
		importer.import(sceneFile, scene);
		app.setScene(scene);
		app.run();
	} catch (std::exception e) {
		logFatal("Kiraray::Unhandled exception: " + string(e.what()));
	}

	return EXIT_SUCCESS;
}

KRR_NAMESPACE_END