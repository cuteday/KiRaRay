#pragma once

#include "imgui.h"
#include "kiraray.h"
#include "io.h"
#include "device/buffer.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

KRR_NAMESPACE_BEGIN

namespace ui = ImGui;
using namespace math;

#ifndef KRR_GL_FUNCS
#define KRR_GL_FUNCS

#    define GL_CHECK( call )                                            \
	do                                                                  \
	  {                                                                 \
		call;                                                           \
		GLenum err = glGetError();                                      \
		if( err != GL_NO_ERROR )                                        \
		  {                                                             \
			std::stringstream ss;                                       \
			ss << "GL error " <<  getGLErrorString( err ) << " at "     \
			   << __FILE__  << "(" <<  __LINE__  << "): " << #call      \
			   << std::endl;                                            \
			std::cerr << ss.str() << std::endl;                         \
			throw std::runtime_error( ss.str().c_str() );               \
		  }                                                             \
	  }                                                                 \
	while (0)


#    define GL_CHECK_ERRORS( )                                          \
	do																	\
	  {                                                                 \
		GLenum err = glGetError();                                      \
		if( err != GL_NO_ERROR )                                        \
		  {                                                             \
			std::stringstream ss;                                       \
			ss << "GL error " <<  getGLErrorString( err ) << " at "     \
			   << __FILE__  << "(" <<  __LINE__  << ")";                \
			std::cerr << ss.str() << std::endl;                         \
			throw std::runtime_error( ss.str().c_str() );               \
		  }                                                             \
	  }                                                                 \
	while (0)

#endif

class WindowApp{
public:
	WindowApp(const char title[], Vector2i size,
			bool visible = true, bool enableVsync = false);

	~WindowApp();

	virtual void resize(const Vector2i &size);

	virtual void run();

	virtual void render() = 0;

	virtual void draw();

	// user input handler
	virtual void onMouseEvent(io::MouseEvent& mouseEvent) {};
	virtual void onKeyEvent(io::KeyboardEvent &keyEvent) {};
	Vector2f getMouseScale() { return fbSize.cast<float>().cwiseInverse(); }

	inline Vector2i getMousePos() const {
		double x, y;
		glfwGetCursorPos(handle, &x, &y);
		return { (int)x, (int)y };
	}
	virtual void renderUI() {};

  protected:
	Vector2i fbSize{0};
	GLuint fbTexture{0};
	GLuint fbPbo{0};
	cudaGraphicsResource_t cuDisplayTexture{0};
	CUDABuffer fbBuffer;

	/*! the glfw window handle */
	GLFWwindow *handle{nullptr};
	Vector2i lastMousePos = {-1, -1};
};

KRR_NAMESPACE_END