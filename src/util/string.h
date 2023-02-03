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

inline string getFileName(string filepath) {
	return std::filesystem::path(filepath).filename().string();
}

inline string getFileNameNoExt(string filepath) {
	return std::filesystem::path(filepath).stem().string();
}

inline std::wstring stringToWideString(const string& src) {
	return std::wstring{src.begin(), src.end()};	
}

KRR_NAMESPACE_END