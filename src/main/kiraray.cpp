#include "kiraray.h"

#include "common.h"
#include "GLFW/glfw3.h"

#include "renderer.h"

NAMESPACE_KRR_BEGIN

extern "C" int main(int argc, char* argv[]) {
	std::cout << KRR_TERMINAL_GREEN 
			"Hello, world!" << 
			KRR_TERMINAL_DEFAULT << std::endl;

	RenderApp app("test renderer", { 1920, 1080 });
	app.run();

	return 0;
}

NAMESPACE_KRR_END