#include "file.h"
#include "config.h"

KRR_NAMESPACE_BEGIN

fs::path File::cwd() { return PROJECT_DIR; }
fs::path File::projectDir() { return cwd(); }
fs::path File::dataDir() { return cwd() / "data"; }
fs::path File::codeDir() { return cwd() / "src"; }

fs::path File::resolve(const fs::path &name) { return File::cwd() / name; }

KRR_NAMESPACE_END