#include "window.h"
#include "logger.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

namespace api {

using namespace krr::io;

inline const char *getGLErrorString(GLenum error) {
	switch (error) {
		case GL_NO_ERROR:
			return "No error";
		case GL_INVALID_ENUM:
			return "Invalid enum";
		case GL_INVALID_VALUE:
			return "Invalid value";
		case GL_INVALID_OPERATION:
			return "Invalid operation";
		case GL_STACK_OVERFLOW:
			return "Stack overflow";
		case GL_STACK_UNDERFLOW:
			return "Stack underflow";
		case GL_OUT_OF_MEMORY:
			return "Out of memory";
		default:
			return "Unknown GL error";
	}
}

void initGLFW() {
	static bool initialized = false;
	if (initialized)
		return;
	if (!glfwInit())
		exit(EXIT_FAILURE);
	initialized = true;
}

void initUI(GLFWwindow *window) {
	static bool initialized = false;
	if (initialized)
		return;
	ui::CreateContext();
	ImGuiIO &io = ui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard 
		|ImGuiConfigFlags_DockingEnable
		|ImGuiConfigFlags_ViewportsEnable;
	ui::StyleColorsLight();
	ImGuiStyle &style = ui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		style.WindowRounding			  = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}
	ImGui_ImplOpenGL3_Init("#version 330");
	ImGui_ImplGlfw_InitForOpenGL(window, true);
}

class ApiCallbacks {
public:
	static void windowSizeCallback(GLFWwindow *pGlfwWindow, int width, int height) {
		// We also get here in case the window was minimized, so we need to
		// ignore it
		if (width * height == 0) {
			return;
		}
		WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
		if (pWindow != nullptr) {
			pWindow->resize({ width, height }); // Window callback is handled in
												// here
		}
	}

	static void keyboardCallback(GLFWwindow *pGlfwWindow, int key, int scanCode, int action,
								 int modifiers) {
		if (ui::GetIO().WantCaptureKeyboard)
			return;
		KeyboardEvent event;
		if (prepareKeyboardEvent(key, action, modifiers, event)) {
			WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
			if (pWindow != nullptr) {
				pWindow->onKeyEvent(event);
			}
		}
	}

	static void charInputCallback(GLFWwindow *pGlfwWindow, uint32_t input) {
		if (ui::GetIO().WantCaptureKeyboard)
			return;
		KeyboardEvent event;
		event.type		= KeyboardEvent::Type::Input;
		event.codepoint = input;

		WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
		if (pWindow != nullptr) {
			pWindow->onKeyEvent(event);
		}
	}

	static void mouseMoveCallback(GLFWwindow *pGlfwWindow, double mouseX, double mouseY) {
		if (ui::GetIO().WantCaptureMouse)
			return;
		WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
		if (pWindow != nullptr) {
			// Prepare the mouse data
			MouseEvent event;
			event.type		 = MouseEvent::Type::Move;
			event.pos		 = calcMousePos(mouseX, mouseY, pWindow->getMouseScale());
			event.screenPos	 = Vector2f(mouseX, mouseY);
			event.wheelDelta = Vector2f(0, 0);

			pWindow->onMouseEvent(event);
		}
	}

	static void mouseButtonCallback(GLFWwindow *pGlfwWindow, int button, int action,
									int modifiers) {
		if (ui::GetIO().WantCaptureMouse)
			return;
		MouseEvent event;
		// Prepare the mouse data
		switch (button) {
			case GLFW_MOUSE_BUTTON_LEFT:
				event.type = (action == GLFW_PRESS) ? MouseEvent::Type::LeftButtonDown
													: MouseEvent::Type::LeftButtonUp;
				break;
			case GLFW_MOUSE_BUTTON_MIDDLE:
				event.type = (action == GLFW_PRESS) ? MouseEvent::Type::MiddleButtonDown
													: MouseEvent::Type::MiddleButtonUp;
				break;
			case GLFW_MOUSE_BUTTON_RIGHT:
				event.type = (action == GLFW_PRESS) ? MouseEvent::Type::RightButtonDown
													: MouseEvent::Type::RightButtonUp;
				break;
			default:
				// Other keys are not supported
				break;
		}

		WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
		if (pWindow != nullptr) {
			// Modifiers
			event.mods = getInputModifiers(modifiers);
			double x, y;
			glfwGetCursorPos(pGlfwWindow, &x, &y);
			event.pos = calcMousePos(x, y, pWindow->getMouseScale());

			pWindow->onMouseEvent(event);
		}
	}

	static void mouseWheelCallback(GLFWwindow *pGlfwWindow, double scrollX, double scrollY) {
		if (ui::GetIO().WantCaptureMouse)
			return;
		WindowApp *pWindow = (WindowApp *) glfwGetWindowUserPointer(pGlfwWindow);
		if (pWindow != nullptr) {
			MouseEvent event;
			event.type = MouseEvent::Type::Wheel;
			double x, y;
			glfwGetCursorPos(pGlfwWindow, &x, &y);
			event.pos		 = calcMousePos(x, y, pWindow->getMouseScale());
			event.wheelDelta = (Vector2f(float(scrollX), float(scrollY)));

			pWindow->onMouseEvent(event);
		}
	}

	static void errorCallback(int errorCode, const char *pDescription) {
		std::string errorMsg = std::to_string(errorCode) + " - " + std::string(pDescription) + "\n";
		logError(errorMsg.c_str());
	}

private:
	static inline InputModifiers getInputModifiers(int mask) {
		InputModifiers mods;
		mods.isAltDown	 = (mask & GLFW_MOD_ALT) != 0;
		mods.isCtrlDown	 = (mask & GLFW_MOD_CONTROL) != 0;
		mods.isShiftDown = (mask & GLFW_MOD_SHIFT) != 0;
		return mods;
	}

	// calculates the mouse pos in sreen [0, 1]^2
	static inline Vector2f calcMousePos(double xPos, double yPos, const Vector2f &mouseScale) {
		Vector2f pos = Vector2f(float(xPos), float(yPos));
		pos		  = pos.cwiseProduct(mouseScale);
		return pos;
	}

	static inline bool prepareKeyboardEvent(int key, int action, int modifiers,
											KeyboardEvent &event) {
		if (action == GLFW_REPEAT || key == GLFW_KEY_UNKNOWN)
			return false;

		event.type = (action == GLFW_RELEASE ? KeyboardEvent::Type::KeyReleased
											 : KeyboardEvent::Type::KeyPressed);
		event.key  = glfwToKey(key);
		event.mods = getInputModifiers(modifiers);
		return true;
	}
};
} // namespace api
using namespace krr::api;

WindowApp::WindowApp(const char title[], Vector2i size, bool visible, bool enableVsync) {
	glfwSetErrorCallback(ApiCallbacks::errorCallback);

	initGLFW();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE,
				   GLFW_OPENGL_COMPAT_PROFILE); // we need this to support
												// immediate mode (for simple
												// framebuffer bliting), instead
												// of core profile only
	glfwWindowHint(GLFW_VISIBLE, visible);

	handle = glfwCreateWindow(size[0], size[1], title, NULL, NULL);
	if (!handle) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(handle, this); // so we can get current "this"
											// pointer in callback
	glfwMakeContextCurrent(handle);
	glfwSwapInterval(enableVsync ? 1 : 0);

	glfwSetWindowSizeCallback(handle, ApiCallbacks::windowSizeCallback);
	glfwSetKeyCallback(handle, ApiCallbacks::keyboardCallback);
	glfwSetMouseButtonCallback(handle, ApiCallbacks::mouseButtonCallback);
	glfwSetCursorPosCallback(handle, ApiCallbacks::mouseMoveCallback);
	glfwSetScrollCallback(handle, ApiCallbacks::mouseWheelCallback);
	glfwSetCharCallback(handle, ApiCallbacks::charInputCallback);

	GLenum e = glewInit();
	if (e != GLEW_OK) {
		glfwDestroyWindow(handle);
		Log(Error, "GLEW initialization failed: %s\n", glewGetErrorString(e));
	}

	initUI(handle);
}

WindowApp::~WindowApp() {
	glfwDestroyWindow(handle);
	glfwTerminate();
}

void WindowApp::resize(const Vector2i size) {
	glfwMakeContextCurrent(handle);
	glfwSetWindowSize(handle, size[0], size[1]);
	fbBuffer.resize(sizeof(Color4f) * size[0] * size[1]);
	fbSize = size;

	if (fbTexture == 0)
		GL_CHECK(glGenTextures(1, &fbTexture));
	if (fbPbo)
		GL_CHECK(glDeleteBuffers(1, &fbPbo));
	if (fbPbo == 0)
		GL_CHECK(glGenBuffers(1, &fbPbo));
	else if (cuDisplayTexture) {
		cudaGraphicsUnregisterResource(cuDisplayTexture);
		cuDisplayTexture = 0;
	}

	GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, fbPbo));
	GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(Color4f) * size[0] * size[1], nullptr,
						  GL_STREAM_DRAW));
	GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

	// We need to re-register when resizing the texture
	cudaError_t rc =
		cudaGraphicsGLRegisterBuffer(&cuDisplayTexture, fbPbo, cudaGraphicsMapFlagsWriteDiscard);

	if (rc != cudaSuccess)
		logError("Could not do CUDA graphics resource sharing (DMA) for the "
				 "display buffer texture (" +
					 string(cudaGetErrorString(cudaGetLastError())) + ")... ",
				 true);
}

void WindowApp::draw() {
	glfwMakeContextCurrent(handle);

	// there are two ways of transfering device data to an opengl texture.
	// via cudaMemcpyToArray, or glTexSubImage2D.
	// the CUDA api is faster, while needing to register the texture to device beforehand.
	CUDA_CHECK(cudaGraphicsMapResources(1, &cuDisplayTexture));

	void *fbPointer;
	size_t fbBytes;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&fbPointer, &fbBytes, cuDisplayTexture));
	CUDA_CHECK(cudaMemcpy(fbPointer, (void *) fbBuffer.data(), fbBytes, cudaMemcpyDeviceToDevice));

	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, fbTexture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fbPbo);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fbSize[0], fbSize[1], 0, GL_RGBA, GL_FLOAT, nullptr);

	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, fbSize[0], fbSize[1]);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, (float) fbSize[0], 0.f, (float) fbSize[1], -1.f, 1.f);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);

		glTexCoord2f(0.f, 1.f);
		glVertex3f(0.f, (float) fbSize[1], 0.f);

		glTexCoord2f(1.f, 1.f);
		glVertex3f((float) fbSize[0], (float) fbSize[1], 0.f);

		glTexCoord2f(1.f, 0.f);
		glVertex3f((float) fbSize[0], 0.f, 0.f);
	}
	glEnd();

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuDisplayTexture));
}

void WindowApp::run() {
	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(Vector2i(width, height));

	while (!glfwWindowShouldClose(handle)) {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ui::NewFrame();

		render();

		draw();
		renderUI();
		{
			PROFILE("Draw UI");
			ui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ui::GetDrawData());
			if (ui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
				ui::UpdatePlatformWindows();
				ui::RenderPlatformWindowsDefault();
			}
		}

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}
}

KRR_NAMESPACE_END