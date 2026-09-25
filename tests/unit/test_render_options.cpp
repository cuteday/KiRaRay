#include <iostream>
#include <limits>
#include <stdexcept>

#include "main/renderoptions.h"

using nlohmann::json;
using krr::RenderOptions;

template <typename F> void rejects(F &&operation) {
	try {
		operation();
	} catch (const std::exception &) {
		return;
	}
	throw std::runtime_error("Invalid render options were accepted");
}

int main() {
	try {
		json config = {{"resolution", {32, 32}}, {"scene", json::object()},
			{"passes", {{{"name", "WavefrontPathTracer"}, {"params", json::object()}}}}};
		RenderOptions::validateConfig(config);
		RenderOptions::validate(1, 0);
		RenderOptions::validate(128, std::numeric_limits<uint64_t>::max());
		rejects([] { RenderOptions::validate(0, 0); });
		rejects([] { RenderOptions::validate(-1, 0); });
		rejects([] { RenderOptions::validate(int64_t(std::numeric_limits<uint32_t>::max()) + 1, 0); });
		rejects([] { RenderOptions::validateConfig(json::array()); });
		rejects([] { RenderOptions::validateConfig(json::object()); });
		for (const json &resolution : {json{0, 32}, json{-1, 32}, json{32}, json{32, 32, 3},
				json{32.5, 32}, json{2147483647, 2147483647}}) {
			json invalid = config;
			invalid["resolution"] = resolution;
			rejects([&] { RenderOptions::validateConfig(invalid); });
		}
		for (const json &passes : {json::array(), json::object(), json::array({json::object()}),
				json::array({{{"name", ""}}}),
				json::array({{{"name", "WavefrontPathTracer"}, {"params", 7}}}),
				json::array({{{"name", "WavefrontPathTracer"}, {"enable", "yes"}}})}) {
			json invalid = config;
			invalid["passes"] = passes;
			rejects([&] { RenderOptions::validateConfig(invalid); });
		}
		json noResolution = config;
		noResolution.erase("resolution");
		RenderOptions::validateConfig(noResolution);
		json noScene = config;
		noScene.erase("scene");
		rejects([&] { RenderOptions::validateConfig(noScene); });
		noScene["model"] = "scene.obj";
		RenderOptions::validateConfig(noScene);
		noScene["model"] = json::array();
		rejects([&] { RenderOptions::validateConfig(noScene); });
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
