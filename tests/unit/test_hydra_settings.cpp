#include "integrations/hydra/settings.h"

#include <iostream>

using krr::hydra::WavefrontSettings;
using nlohmann::json;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

int main() {
	try {
		WavefrontSettings settings;
		require(settings.params() == json{{"max_depth", 10}, {"nee", true}, {"rr", 0.8f}},
			"Hydra wavefront defaults changed");
		for (const auto &[name, value] : std::vector<std::pair<std::string, json>>{
			 {"krr:maxDepth", -1}, {"krr:maxDepth", 1.5}, {"krr:maxDepth", true},
			 {"krr:maxDepth", uint64_t(std::numeric_limits<int>::max()) + 1},
			 {"krr:nee", 1}, {"krr:nee", "true"}, {"krr:rr", 0}, {"krr:rr", -0.1},
			 {"krr:rr", 1.1}, {"krr:rr", 1.00000001}, {"krr:rr", 1e-300},
			 {"krr:rr", 1e300}, {"krr:rr", -1e300},
			 {"krr:rr", true}, {"krr:rr", "0.8"},
			 {"krr:rr", std::numeric_limits<float>::infinity()},
			 {"krr:rr", std::numeric_limits<float>::quiet_NaN()}}) {
			const auto before = settings.params();
			bool rejected = false;
			try { settings.set(name, value); }
			catch (const std::invalid_argument &) { rejected = true; }
			require(rejected, "Invalid Hydra wavefront setting was accepted");
			require(settings.params() == before, "Rejected setting changed the active configuration");
		}
		require(settings.set("krr:maxDepth", 0), "Zero-depth setting was not recognized");
		require(settings.set("krr:nee", false), "NEE setting was not recognized");
		require(settings.set("krr:rr", 1.0), "Unit survival probability was rejected");
		require(settings.params() == json{{"max_depth", 0}, {"nee", false}, {"rr", 1.f}},
			"Settings do not forward the native wavefront parameter names");
		settings.set("krr:maxDepth", std::numeric_limits<int>::max());
		require(settings.maxDepth == std::numeric_limits<int>::max(), "Unexpected hard path-depth limit");
		require(!settings.set("krr:samples", 128), "Unrelated settings were consumed");
		settings.set("krr:maxDepth", 10);
		settings.set("krr:nee", true);
		settings.set("krr:rr", 0.8);
		require(settings.params() == WavefrontSettings{}.params(), "Updating settings did not restore defaults");
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
