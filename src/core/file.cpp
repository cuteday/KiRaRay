#include "file.h"
#include "config.h"

KRR_NAMESPACE_BEGIN

fs::path File::cwd() { return KRR_PROJECT_DIR; }
fs::path File::projectDir() { return cwd(); }
fs::path File::dataDir() { return fs::weakly_canonical(cwd() / "common"); }
fs::path File::codeDir() { return fs::weakly_canonical(cwd() / "src"); }

fs::path File::resolve(const fs::path &name) { 
	return name.is_absolute() ? name : fs::weakly_canonical(File::cwd() / name);
}

KRR_NAMESPACE_END