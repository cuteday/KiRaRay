#include "window.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "gpu/optix7.h"

KRR_NAMESPACE_BEGIN

namespace api{

	using namespace krr::io;
    inline KeyboardEvent::Key glfwToFalcorKey(int glfwKey);

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

	class ApiCallbacks
    {
    public:
        static void windowSizeCallback(GLFWwindow* pGlfwWindow, int width, int height){
            // We also get here in case the window was minimized, so we need to ignore it
            if (width * height == 0){
                return;
            }
            WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr){
                pWindow->resize({width, height}); // Window callback is handled in here
            }
        }

        static void keyboardCallback(GLFWwindow* pGlfwWindow, int key, int scanCode, int action, int modifiers){
			if (ImGui::GetIO().WantCaptureKeyboard) return;
            KeyboardEvent event;
            if (prepareKeyboardEvent(key, action, modifiers, event)){
                WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
                if (pWindow != nullptr){
                    pWindow->onKeyEvent(event);
                }
            }
        }

        static void charInputCallback(GLFWwindow* pGlfwWindow, uint32_t input){
			if (ImGui::GetIO().WantCaptureKeyboard) return;
            KeyboardEvent event;
            event.type = KeyboardEvent::Type::Input;
            event.codepoint = input;

            WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr){
                pWindow->onKeyEvent(event);
            }
        }

        static void mouseMoveCallback(GLFWwindow* pGlfwWindow, double mouseX, double mouseY){
			if (ImGui::GetIO().WantCaptureMouse) return;
            WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr){
                // Prepare the mouse data
                MouseEvent event;
                event.type = MouseEvent::Type::Move;
                event.pos = calcMousePos(mouseX, mouseY, pWindow->getMouseScale());
                event.screenPos = vec2f(mouseX, mouseY);
                event.wheelDelta = vec2f(0, 0);

                pWindow->onMouseEvent(event);
            }
        }

        static void mouseButtonCallback(GLFWwindow* pGlfwWindow, int button, int action, int modifiers){
			if (ImGui::GetIO().WantCaptureMouse) return;
            MouseEvent event;
            // Prepare the mouse data
            switch (button){
            case GLFW_MOUSE_BUTTON_LEFT:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::LeftButtonDown : MouseEvent::Type::LeftButtonUp;
                break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::MiddleButtonDown : MouseEvent::Type::MiddleButtonUp;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::RightButtonDown : MouseEvent::Type::RightButtonUp;
                break;
            default:
                // Other keys are not supported
                break;
            }

            WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr){
                // Modifiers
                event.mods = getInputModifiers(modifiers);
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());

                pWindow->onMouseEvent(event);
            }
        }

        static void mouseWheelCallback(GLFWwindow* pGlfwWindow, double scrollX, double scrollY){
			if (ImGui::GetIO().WantCaptureMouse) return;
            WindowApp* pWindow = (WindowApp*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr){
                MouseEvent event;
                event.type = MouseEvent::Type::Wheel;
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());
                event.wheelDelta = (vec2f(float(scrollX), float(scrollY)));

                pWindow->onMouseEvent(event);
            }
        }

        static void errorCallback(int errorCode, const char* pDescription){
            std::string errorMsg = std::to_string(errorCode) + " - " + std::string(pDescription) + "\n";
            logError(errorMsg.c_str());
        }

    private:

        static inline InputModifiers getInputModifiers(int mask){
            InputModifiers mods;
            mods.isAltDown = (mask & GLFW_MOD_ALT) != 0;
            mods.isCtrlDown = (mask & GLFW_MOD_CONTROL) != 0;
            mods.isShiftDown = (mask & GLFW_MOD_SHIFT) != 0;
            return mods;
        }

		// calculates the mouse pos in sreen [0, 1]^2
        static inline vec2f calcMousePos(double xPos, double yPos, const vec2f& mouseScale){
            vec2f pos = vec2f(float(xPos), float(yPos));
            pos *= mouseScale;
            return pos;
        }

        static inline bool prepareKeyboardEvent(int key, int action, int modifiers, KeyboardEvent& event){
            if (action == GLFW_REPEAT || key == GLFW_KEY_UNKNOWN){
                return false;
            }

            event.type = (action == GLFW_RELEASE ? KeyboardEvent::Type::KeyReleased : KeyboardEvent::Type::KeyPressed);
            event.key = glfwToFalcorKey(key);
            event.mods = getInputModifiers(modifiers);
            return true;
        }
    };
}
using namespace krr::api;

WindowApp::WindowApp(const char title[], vec2i size, bool visible, bool enableVsync)
{
	//glfwSetErrorCallback(glfw_error_callback);
	glfwSetErrorCallback(ApiCallbacks::errorCallback);

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

	glfwSetWindowSizeCallback(handle, ApiCallbacks::windowSizeCallback);
	glfwSetKeyCallback(handle, ApiCallbacks::keyboardCallback);
	glfwSetMouseButtonCallback(handle, ApiCallbacks::mouseButtonCallback);
	glfwSetCursorPosCallback(handle, ApiCallbacks::mouseMoveCallback);
	glfwSetScrollCallback(handle, ApiCallbacks::mouseWheelCallback);
	glfwSetCharCallback(handle, ApiCallbacks::charInputCallback);

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

		renderUI();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}
}

namespace api {
    inline KeyboardEvent::Key glfwToFalcorKey(int glfwKey)
    {
        static_assert(GLFW_KEY_ESCAPE == 256, "GLFW_KEY_ESCAPE is expected to be 256");
        if (glfwKey < GLFW_KEY_ESCAPE)
        {
            // Printable keys are expected to have the same value
            return (KeyboardEvent::Key)glfwKey;
        }

        switch (glfwKey)
        {
        case GLFW_KEY_ESCAPE:
            return KeyboardEvent::Key::Escape;
        case GLFW_KEY_ENTER:
            return KeyboardEvent::Key::Enter;
        case GLFW_KEY_TAB:
            return KeyboardEvent::Key::Tab;
        case GLFW_KEY_BACKSPACE:
            return KeyboardEvent::Key::Backspace;
        case GLFW_KEY_INSERT:
            return KeyboardEvent::Key::Insert;
        case GLFW_KEY_DELETE:
            return KeyboardEvent::Key::Del;
        case GLFW_KEY_RIGHT:
            return KeyboardEvent::Key::Right;
        case GLFW_KEY_LEFT:
            return KeyboardEvent::Key::Left;
        case GLFW_KEY_DOWN:
            return KeyboardEvent::Key::Down;
        case GLFW_KEY_UP:
            return KeyboardEvent::Key::Up;
        case GLFW_KEY_PAGE_UP:
            return KeyboardEvent::Key::PageUp;
        case GLFW_KEY_PAGE_DOWN:
            return KeyboardEvent::Key::PageDown;
        case GLFW_KEY_HOME:
            return KeyboardEvent::Key::Home;
        case GLFW_KEY_END:
            return KeyboardEvent::Key::End;
        case GLFW_KEY_CAPS_LOCK:
            return KeyboardEvent::Key::CapsLock;
        case GLFW_KEY_SCROLL_LOCK:
            return KeyboardEvent::Key::ScrollLock;
        case GLFW_KEY_NUM_LOCK:
            return KeyboardEvent::Key::NumLock;
        case GLFW_KEY_PRINT_SCREEN:
            return KeyboardEvent::Key::PrintScreen;
        case GLFW_KEY_PAUSE:
            return KeyboardEvent::Key::Pause;
        case GLFW_KEY_F1:
            return KeyboardEvent::Key::F1;
        case GLFW_KEY_F2:
            return KeyboardEvent::Key::F2;
        case GLFW_KEY_F3:
            return KeyboardEvent::Key::F3;
        case GLFW_KEY_F4:
            return KeyboardEvent::Key::F4;
        case GLFW_KEY_F5:
            return KeyboardEvent::Key::F5;
        case GLFW_KEY_F6:
            return KeyboardEvent::Key::F6;
        case GLFW_KEY_F7:
            return KeyboardEvent::Key::F7;
        case GLFW_KEY_F8:
            return KeyboardEvent::Key::F8;
        case GLFW_KEY_F9:
            return KeyboardEvent::Key::F9;
        case GLFW_KEY_F10:
            return KeyboardEvent::Key::F10;
        case GLFW_KEY_F11:
            return KeyboardEvent::Key::F11;
        case GLFW_KEY_F12:
            return KeyboardEvent::Key::F12;
        case GLFW_KEY_KP_0:
            return KeyboardEvent::Key::Keypad0;
        case GLFW_KEY_KP_1:
            return KeyboardEvent::Key::Keypad1;
        case GLFW_KEY_KP_2:
            return KeyboardEvent::Key::Keypad2;
        case GLFW_KEY_KP_3:
            return KeyboardEvent::Key::Keypad3;
        case GLFW_KEY_KP_4:
            return KeyboardEvent::Key::Keypad4;
        case GLFW_KEY_KP_5:
            return KeyboardEvent::Key::Keypad5;
        case GLFW_KEY_KP_6:
            return KeyboardEvent::Key::Keypad6;
        case GLFW_KEY_KP_7:
            return KeyboardEvent::Key::Keypad7;
        case GLFW_KEY_KP_8:
            return KeyboardEvent::Key::Keypad8;
        case GLFW_KEY_KP_9:
            return KeyboardEvent::Key::Keypad9;
        case GLFW_KEY_KP_DECIMAL:
            return KeyboardEvent::Key::KeypadDel;
        case GLFW_KEY_KP_DIVIDE:
            return KeyboardEvent::Key::KeypadDivide;
        case GLFW_KEY_KP_MULTIPLY:
            return KeyboardEvent::Key::KeypadMultiply;
        case GLFW_KEY_KP_SUBTRACT:
            return KeyboardEvent::Key::KeypadSubtract;
        case GLFW_KEY_KP_ADD:
            return KeyboardEvent::Key::KeypadAdd;
        case GLFW_KEY_KP_ENTER:
            return KeyboardEvent::Key::KeypadEnter;
        case GLFW_KEY_KP_EQUAL:
            return KeyboardEvent::Key::KeypadEqual;
        case GLFW_KEY_LEFT_SHIFT:
            return KeyboardEvent::Key::LeftShift;
        case GLFW_KEY_LEFT_CONTROL:
            return KeyboardEvent::Key::LeftControl;
        case GLFW_KEY_LEFT_ALT:
            return KeyboardEvent::Key::LeftAlt;
        case GLFW_KEY_LEFT_SUPER:
            return KeyboardEvent::Key::LeftSuper;
        case GLFW_KEY_RIGHT_SHIFT:
            return KeyboardEvent::Key::RightShift;
        case GLFW_KEY_RIGHT_CONTROL:
            return KeyboardEvent::Key::RightControl;
        case GLFW_KEY_RIGHT_ALT:
            return KeyboardEvent::Key::RightAlt;
        case GLFW_KEY_RIGHT_SUPER:
            return KeyboardEvent::Key::RightSuper;
        case GLFW_KEY_MENU:
            return KeyboardEvent::Key::Menu;
        default:
            KRR_SHOULDNT_GO_HERE;
            return (KeyboardEvent::Key)0;
        }
    }

}

KRR_NAMESPACE_END