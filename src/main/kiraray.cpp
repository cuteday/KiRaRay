#include "renderer.h"

NAMESPACE_BEGIN(krr)

extern "C" int main(int argc, char *argv[]) {
	gpContext = std::make_unique<Context>();
	fs::current_path(File::cwd());
	Log(Info, "Working directory: %s", KRR_PROJECT_DIR);
	Log(Info, "Kiraray build type: %s", KRR_BUILD_TYPE);
#ifdef KRR_DEBUG_BUILD
	Log(Warning, "Running in debug mode, the performance may be extremely slow. "
			   "Switch to Release build for normal performance!");
#endif

	string configFile = "common/configs/example_cbox.json";
	if (argc < 2){
		Log(Warning, "No config file specified, using default config file: %s", configFile.c_str());
	} else {
		configFile = argv[1];
		Log(Info, "Using specified config file at %s", configFile.c_str());
	}

	auto app = std::make_shared<RenderApp>();
	try {
		app->loadConfigFrom(configFile);
		app->run();
	} catch (const std::exception &e) {
		Log(Error, "Error: %s", e.what());
	}

	exit(EXIT_SUCCESS);
}

NAMESPACE_END(krr)