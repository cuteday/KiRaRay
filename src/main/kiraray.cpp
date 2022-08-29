#include <filesystem>

#include "file.h"
#include "kiraray.h"

#include "main/renderer.h"
#include "render/passes/accumulate.h"
#include "render/passes/tonemapping.h"
#include "render/passes/denoise.h"
#include "render/path/pathtracer.h"
#include "render/wavefront/integrator.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char *argv[]) {
    std::filesystem::current_path(File::cwd());
    Log(Info, "Working directory: %s\n", KRR_PROJECT_DIR);
	Log(Info, "Kiraray build type: %s", KRR_BUILD_TYPE);
#ifdef KRR_DEBUG_BUILD
	Log(Warning, "Running in debug mode, the performance may be extremely slow. "
               "Switch to Release build for normal performance!");
#endif

    string configFile = "common/configs/cbox.json";
    if (argc < 2){
	    Log(Warning, "No config file specified, using default config file: %s", configFile.c_str());
    } else {
        configFile = argv[1];
        Log(Info, "Using specified config file at %s", configFile.c_str());
    }

	try {
        gpContext = Context::SharedPtr(new Context());
		RenderApp app(KRR_PROJECT_NAME, { 1280, 720 },
					  { // RenderPass::SharedPtr(new PathTracer()),
						RenderPass::SharedPtr(new WavefrontPathTracer()),
						RenderPass::SharedPtr(new AccumulatePass()),
						RenderPass::SharedPtr(new DenoisePass(false)),
						RenderPass::SharedPtr(new ToneMappingPass())});
		app.loadConfig(configFile);
		app.run();
    } catch (std::exception e) {
        Log(Fatal, "Kiraray::Unhandled exception: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}

KRR_NAMESPACE_END