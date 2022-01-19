#include "window.h"

NAMESPACE_KRR_BEGIN

inline const char* getGLErrorString(GLenum error)
{
  switch (error)
	{
	case GL_NO_ERROR:            return "No error";
	case GL_INVALID_ENUM:        return "Invalid enum";
	case GL_INVALID_VALUE:       return "Invalid value";
	case GL_INVALID_OPERATION:   return "Invalid operation";
	case GL_STACK_OVERFLOW:      return "Stack overflow";
	case GL_STACK_UNDERFLOW:     return "Stack underflow";
	case GL_OUT_OF_MEMORY:       return "Out of memory";
	default:                     return "Unknown GL error";
	}
}

void initGLFW()
{
	static bool alreadyInitialized = false;
	if (alreadyInitialized) return;
	if (!glfwInit())
		exit(EXIT_FAILURE);
	alreadyInitialized = true;
}

static void glfw_error_callback(int error, const char *description)
{
	fprintf(stderr, "Error: %s\n", description);
}

WindowApp::WindowApp(const char title[], vec2i size, bool visible, bool enableVsync)
{
	glfwSetErrorCallback(glfw_error_callback);

	initGLFW();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, visible);

	handle = glfwCreateWindow(size.x, size.y, title, NULL, NULL);
	if (!handle)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(handle, this);
	glfwMakeContextCurrent(handle);
	glfwSwapInterval((enableVsync) ? 1 : 0);
}

WindowApp::~WindowApp(){
	glfwDestroyWindow(handle);
	glfwTerminate();
}

void WindowApp::resize(const vec2i &size)
{
	glfwMakeContextCurrent(handle);
	if (fbPointer)
		cudaFree(fbPointer);
	cudaMallocManaged(&fbPointer, size.x * size.y * sizeof(uint32_t));

	fbSize = size;
	if (fbTexture == 0)
	{
		GL_CHECK(glGenTextures(1, &fbTexture));
	}
	else if (cuDisplayTexture)
	{
		cudaGraphicsUnregisterResource(cuDisplayTexture);
		cuDisplayTexture = 0;
	}

	GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
	GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA,
						  GL_UNSIGNED_BYTE, nullptr));

	// We need to re-register when resizing the texture
	cudaError_t rc = cudaGraphicsGLRegisterImage(&cuDisplayTexture, fbTexture, GL_TEXTURE_2D, 0);

	bool forceSlowDisplay = false;
	if (rc != cudaSuccess || forceSlowDisplay)
	{
		std::cout << KRR_TERMINAL_RED
				  << "Warning: Could not do CUDA graphics resource sharing "
				  << "for the display buffer texture ("
				  << cudaGetErrorString(cudaGetLastError())
				  << ")... falling back to slower path"
				  << KRR_TERMINAL_DEFAULT
				  << std::endl;
		resourceSharingSuccessful = false;
		if (cuDisplayTexture)
		{
			cudaGraphicsUnregisterResource(cuDisplayTexture);
			cuDisplayTexture = 0;
		}
	}
	else
	{
		resourceSharingSuccessful = true;
	}
	//setAspect(fbSize.x/float(fbSize.y));
}

void WindowApp::draw()
{
	glfwMakeContextCurrent(handle);
	if (resourceSharingSuccessful)
	{
		GL_CHECK(cudaGraphicsMapResources(1, &cuDisplayTexture));

		cudaArray_t array;
		// render data are put into fbPointer, which is mapped onto fbTexture
		GL_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));
		{
			cudaMemcpy2DToArray(array,
								0,
								0,
								reinterpret_cast<const void *>(fbPointer),
								fbSize.x * sizeof(uint32_t),
								fbSize.x * sizeof(uint32_t),
								fbSize.y,
								cudaMemcpyDeviceToDevice);
		}
	}
	else
	{
		// if cuda graphics interop failed, use opengl api to copy data into fbTexture
		GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
		glEnable(GL_TEXTURE_2D);
		GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0,
								 0, 0,
								 fbSize.x, fbSize.y,
								 GL_RGBA, GL_UNSIGNED_BYTE, fbPointer));
	}
	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, fbTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, fbSize.x, fbSize.y);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);

		glTexCoord2f(0.f, 1.f);
		glVertex3f(0.f, (float)fbSize.y, 0.f);

		glTexCoord2f(1.f, 1.f);
		glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

		glTexCoord2f(1.f, 0.f);
		glVertex3f((float)fbSize.x, 0.f, 0.f);
	}
	glEnd();

	if (resourceSharingSuccessful)
	{
		GL_CHECK(cudaGraphicsUnmapResources(1, &cuDisplayTexture));
	}
}

void WindowApp::run()
{
	int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width,height));

	while (!glfwWindowShouldClose(handle))
	{
		render();
		draw();

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}

	glfwDestroyWindow(handle);
	glfwTerminate();
}

NAMESPACE_KRR_END