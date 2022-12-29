#include "file.h"
#include "config.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN

fs::path File::m_current_working_dir{ KRR_PROJECT_DIR };
fs::path File::m_output_dir{ KRR_PROJECT_DIR };

fs::path File::cwd() { return m_current_working_dir; }
fs::path File::outputDir() { return m_output_dir; }

fs::path File::projectDir() { return cwd(); }
fs::path File::dataDir() { return fs::weakly_canonical(cwd() / "common"); }
fs::path File::codeDir() { return fs::weakly_canonical(cwd() / "src"); }

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

KRR_NAMESPACE_END