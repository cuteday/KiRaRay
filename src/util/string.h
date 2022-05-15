#pragma once

#include "common.h"
#include <filesystem>

KRR_NAMESPACE_BEGIN

// returns file extension with prefix '.'
inline string getFileExt(string filepath) {
	return std::filesystem::path(filepath).extension().string();
}

inline string getFileDir(string filepath) {
	return std::filesystem::path(filepath).parent_path().string();
}

KRR_NAMESPACE_END