#pragma once

#include <filesystem>
#include "common.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;

class Blob {
public:
	Blob(void *data, size_t size) : m_data(data), m_size(size) {}
	~Blob() { if (m_data) free(m_data); m_data = nullptr;};
	const void *data() const { return m_data; }
	size_t size() const { return m_size; }

private:
	void *m_data;
	size_t m_size;
};

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
	static void saveJSON(const fs::path &filepath, const json &j);

	static std::shared_ptr<Blob> readFile(const fs::path &filepath, bool binary=true);

private:
	static fs::path m_current_working_dir;
	static fs::path m_output_dir;
};

KRR_NAMESPACE_END