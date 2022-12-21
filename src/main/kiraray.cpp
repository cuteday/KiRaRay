#include <filesystem>

#include "file.h"
#include "kiraray.h"
#include "renderer.h"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char *argv[]) {
    fs::current_path(File::cwd());
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
        gpContext = std::make_shared<Context>();
		RenderApp app(KRR_PROJECT_NAME);
		app.loadConfig(configFile);
		app.run();
    } catch (std::exception e) {
        Log(Fatal, "Kiraray::Unhandled exception: %s\n", e.what());
    }

    exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END