#include <iostream>
#include <stdexcept>

#include "scene/importer.h"

using namespace krr;

template <typename F> void requireUnavailable(F load) {
	try {
		load();
	} catch (const std::runtime_error &error) {
		if (std::string(error.what()).find("KRR_ENABLE_OPENVDB_IO=ON") != std::string::npos)
			return;
		throw;
	}
	throw std::runtime_error("Disabled volume import did not report the required build option");
}

int main() {
	try {
		for (const char *path : {"missing.vdb", "missing.nvdb"}) {
			requireUnavailable([&] { SceneImporter::loadModel(path, nullptr); });
			requireUnavailable([&] {
				SceneImporter::loadMedium(nullptr, nullptr,
					json{{"type", "heterogeneous"}, {"file", path}});
			});
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
