#pragma once

#include "common.h"
#include <filesystem>
#include <stdexcept>

NAMESPACE_BEGIN(krr)

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

template <typename... Args> 
inline std::string formatString(const std::string &format, Args&& ...args) {
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size_s <= 0) throw std::runtime_error("Error during formatting.");
	auto size = static_cast<size_t>(size_s);
	auto buf = std::make_unique<char[]>(size);
	std::snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

NAMESPACE_END(krr)