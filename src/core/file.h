#pragma once

#include <filesystem>
#include "common.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;

class File {
public:
	static fs::path cwd();
	static fs::path projectDir();

	static fs::path outputDir();
	static fs::path dataDir();
	static fs::path codeDir();

	static void setOutputDir(const fs::path &outputDir);
	static void setCwd(const fs::path &cwd);

	static fs::path resolve(const fs::path &name);
	static json loadJSON(const fs::path &filepath);

private:
	static fs::path m_current_working_dir;
	static fs::path m_output_dir;
};

KRR_NAMESPACE_END