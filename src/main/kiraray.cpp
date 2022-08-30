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

void registerRenderPasses() {
    // just a temporary workaroud...
	RenderPass::SharedPtr __MegakernelPathTracer(new MegakernelPathTracer());
	RenderPass::SharedPtr __WavefrontPathTracer(new WavefrontPathTracer());
	RenderPass::SharedPtr __AccumulatePass(new AccumulatePass());
	RenderPass::SharedPtr __DenoisePass(new DenoisePass());
	RenderPass::SharedPtr __ToneMappingPass(new ToneMappingPass());
}

extern "C" int main(int argc, char *argv[]) {
    std::filesystem::current_path(File::cwd());
    Log(Info, "Working directory: %s\n", KRR_PROJECT_DIR);
	Log(Info, "Kiraray build type: %s", KRR_BUILD_TYPE);
#ifdef KRR_DEBUG_BUILD
	Log(Warning, "Running in debug mode, the performance may be extremely slow. "
               "Switch to Release build for normal performance!");
#endif

    string configFile = "common/configs/example.json";
    if (argc < 2){
	    Log(Warning, "No config file specified, using default config file: %s", configFile.c_str());
    } else {
        configFile = argv[1];
        Log(Info, "Using specified config file at %s", configFile.c_str());
    }

	try {
        gpContext = Context::SharedPtr(new Context());
		registerRenderPasses();
		RenderApp app(KRR_PROJECT_NAME);
		app.loadConfig(configFile);
		app.run();
    } catch (std::exception e) {
        Log(Fatal, "Kiraray::Unhandled exception: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}

KRR_NAMESPACE_END