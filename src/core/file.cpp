#include <fstream>

#include "file.h"
#include "config.h"
#include "logger.h"

NAMESPACE_BEGIN(krr)

fs::path File::m_current_working_dir{ KRR_PROJECT_DIR };
fs::path File::m_output_dir{ KRR_PROJECT_DIR };

fs::path File::cwd() { return m_current_working_dir; }
fs::path File::outputDir() { return m_output_dir; }

fs::path File::projectDir() { return cwd(); }
fs::path File::dataDir() { return fs::weakly_canonical(cwd() / "common"); }
fs::path File::codeDir() { return fs::weakly_canonical(cwd() / "src"); }
fs::path File::assetDir() { return fs::weakly_canonical(dataDir() / "assets"); }
fs::path File::textureDir() { return fs::weakly_canonical(assetDir() / "textures"); }

fs::path File::resolve(const fs::path &name) { 
	return name.is_absolute() ? name : fs::weakly_canonical(File::cwd() / name);
}

void File::setOutputDir(const fs::path& outputDir) {
	Log(Info, "Setting output directory to %s", outputDir.string().c_str());
	if (!fs::exists(outputDir)) {
		fs::create_directories(outputDir);
	} else if (fs::exists(outputDir) && !fs::is_directory(outputDir)) {
		Log(Error, "%s is not a directory!", outputDir.string().c_str());
		return;
	}
	m_output_dir = outputDir;
}

void File::setCwd(const fs::path &cwd) {
	Log(Info, "Setting working directory to %s", cwd.string().c_str());
	if (!fs::exists(cwd)) {
		fs::create_directories(cwd);
	} else if (fs::exists(cwd) && !fs::is_directory(cwd)) {
		Log(Error, "%s is not a directory!", cwd.string().c_str());
		return;
	}
	m_current_working_dir = cwd;
}

json File::loadJSON(const fs::path& filepath) {
	if (!fs::exists(filepath)) {
		Log(Error, "Cannot locate file at %s", filepath.string().c_str());
		return {};
	}
	std::ifstream f(filepath);
	if (f.fail()) {
		Log(Error, "Failed to read JSON file at %s", filepath.string().c_str());
		return {};
	}
	json file = json::parse(f, nullptr, true, true);
	return file;
}

void File::saveJSON(const fs::path &filepath, const json &j) {
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	std::ofstream ofs(filepath);
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
}

std::shared_ptr<Blob> File::readFile(const fs::path &filepath, bool binary) { 
	auto flags = binary ? (std::ios::binary | std::ios::in) : std::ios::in; 
	std::ifstream s(filepath, flags);

	if (!s.is_open()) {
		// file does not exist or is locked
		return nullptr;
	}
	s.seekg(0, std::ios_base::end);
	size_t size = s.tellg();
	s.clear();
	s.seekg(0, std::ios_base::beg);

	if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
		// file larger than size_t
		assert(false);
		return nullptr;
	}

	// TODO: check why malloc causes error here...
	char *data = static_cast<char *>(calloc(size, sizeof(char)));

	if (data == nullptr) {
		// out of memory
		assert(false);
		return nullptr;
	}
	s.read(data, size);

	// TODO: check why some text file introduces this error...
	//if (!s.good()) {
	//	// reading error
	//	assert(false);
	//	return nullptr;
	//}
	return std::make_shared<Blob>(data, size);
}

NAMESPACE_END(krr)