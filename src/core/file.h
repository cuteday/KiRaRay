#pragma once

#include <filesystem>
#include "common.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;

namespace File{
	fs::path cwd();
	fs::path projectDir();
	fs::path dataDir();
	fs::path codeDir();

	fs::path resolve(const fs::path &name);
}

KRR_NAMESPACE_END