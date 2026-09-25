#include "renderer.h"

NAMESPACE_BEGIN(krr)

extern "C" int main(int argc, char *argv[]) {
	int result = EXIT_SUCCESS;
	try {
		Log(Info, "Asset directory: %s", KRR_PROJECT_DIR);
		Log(Info, "Kiraray build type: %s", KRR_BUILD_TYPE);
#ifdef KRR_DEBUG_BUILD
		Log(Warning, "Running in debug mode, the performance may be slow."
			"Switch to Release build for faster performance!");
#endif

		string configFile = "common/configs/example_cbox.json";
		if (argc < 2) {
			Log(Warning, "No config file specified, using default config file: %s", configFile.c_str());
		} else {
			configFile = argv[1];
			Log(Info, "Using specified config file at %s", configFile.c_str());
		}

		RenderApp app;
		app.loadConfigFrom(File::resolve(configFile));
		app.run();
	} catch (const std::exception &e) {
		Log(Error, "Error: %s", e.what());
		result = EXIT_FAILURE;
	}
	gpContext.reset();
	return result;
}

NAMESPACE_END(krr)
