#pragma once

#include "kiraray.h"

#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "math/vec.h"

NAMESPACE_KRR_BEGIN

using namespace math;

class WindowApp{
private:

	vec2i fbSize{0};
	GLuint   fbTexture  {0};
	cudaGraphicsResource_t cuDisplayTexture { 0 };
	uint32_t *fbPointer { nullptr };
	
	/*! the glfw window handle */
	GLFWwindow *handle { nullptr };

};

NAMESPACE_KRR_END