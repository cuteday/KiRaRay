#include "window.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "gpu/optix7.h"

KRR_NAMESPACE_BEGIN

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
	static bool initialized = false;
	if (initialized) return;
	if (!glfwInit())
		exit(EXIT_FAILURE);
	initialized = true;
}

void initImGui(GLFWwindow* window){
	static bool initialized = false;
	if(initialized) return;
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsLight();
    ImGui_ImplOpenGL3_Init("#version 150"); // Mac compatible: GL 3.2 + GLSL 150
	ImGui_ImplGlfw_InitForOpenGL(window, true);
}

// glfw callback interface
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (ImGui::GetIO().WantCaptureKeyboard) return;
	WindowApp* app = static_cast<WindowApp*>(glfwGetWindowUserPointer(window));
	assert(app);
	if (action == GLFW_PRESS) {
		app->key(key, mods);
	}
}

static void glfw_mouseMotion_callback(GLFWwindow* window, double x, double y)
{
	if (ImGui::GetIO().WantCaptureMouse) return;
	WindowApp* app = static_cast<WindowApp*>(glfwGetWindowUserPointer(window));
	assert(app);
	app->mouseMotion(vec2i((int)x, (int)y));
}

static void glfw_mouseButton_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (ImGui::GetIO().WantCaptureMouse) return;
	WindowApp* app = static_cast<WindowApp*>(glfwGetWindowUserPointer(window));
	assert(app);
	app->mouseButton(button, action, mods);
}

static void glfw_error_callback(int error, const char *description)
{
	logError("GLFW Error: %s\n" + string(description));
}

static void glfw_resize_callback(GLFWwindow* window, int width, int height) {
	WindowApp* app = static_cast<WindowApp*>(glfwGetWindowUserPointer(window));
	app->resize({ width, height });
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

	glfwSetWindowUserPointer(handle, this); // so we can get current "this" pointer in callback
	glfwMakeContextCurrent(handle);
	glfwSwapInterval((enableVsync) ? 1 : 0);

	glfwSetFramebufferSizeCallback(handle, glfw_resize_callback);
	glfwSetMouseButtonCallback(handle, glfw_mouseButton_callback);
	glfwSetKeyCallback(handle, glfw_key_callback);
	glfwSetCursorPosCallback(handle, glfw_mouseMotion_callback);

	initImGui(handle);
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
	cudaMallocManaged(&fbPointer, sizeof(vec4f) * size.x * size.y);

	fbSize = size;
	if (fbTexture == 0)
		GL_CHECK(glGenTextures(1, &fbTexture));
	else if (cuDisplayTexture)
	{
		cudaGraphicsUnregisterResource(cuDisplayTexture);
		cuDisplayTexture = 0;
	}

	GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
	GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA,
						  GL_FLOAT, nullptr));

	// We need to re-register when resizing the texture
	cudaError_t rc = cudaGraphicsGLRegisterImage(&cuDisplayTexture, fbTexture, GL_TEXTURE_2D, 0);

	bool forceSlowDisplay = false;
	if (rc != cudaSuccess || forceSlowDisplay)
	{
		logError("Could not do CUDA graphics resource sharing "
			"for the display buffer texture (" + 
			string(cudaGetErrorString(cudaGetLastError())) +
			")... falling back to slower path"
			);

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
		CUDA_CHECK(cudaGraphicsMapResources(1, &cuDisplayTexture));

		cudaArray_t array;
		// render data are put into fbPointer, which is mapped onto fbTexture
		CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));

		CUDA_CHECK(cudaMemcpy2DToArray(array,
								0,
								0,
								reinterpret_cast<const void*>(fbPointer),
								fbSize.x * sizeof(vec4f),
								fbSize.x * sizeof(vec4f),
								fbSize.y,
								cudaMemcpyDeviceToDevice));
		
		//CUDA_CHECK(cudaMemcpyToArray(array, 0, 0, reinterpret_cast<const void*>(fbPointer),
		//	sizeof(vec4f) * fbSize.x * fbSize.y, cudaMemcpyDeviceToDevice));
	}
	else
	{
		// if cuda graphics interop failed, use opengl api to copy data into fbTexture
		GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
		glEnable(GL_TEXTURE_2D);
		GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0,
								 0, 0,
								 fbSize.x, fbSize.y,
								 GL_RGBA, GL_FLOAT, fbPointer));
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
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuDisplayTexture));
	}
}

void WindowApp::run()
{
	int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width,height));

	while (!glfwWindowShouldClose(handle))
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render();
		draw();

		if (renderUI) draw_ui();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}

	glfwDestroyWindow(handle);
	glfwTerminate();
}

KRR_NAMESPACE_END