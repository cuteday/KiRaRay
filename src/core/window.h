#pragma once

#include "kiraray.h"

#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "math/vec.h"

NAMESPACE_KRR_BEGIN

using namespace math;

#ifndef KRR_GL_FUNCS
#define KRR_GL_FUNCS

inline const char* getGLErrorString( GLenum error )
    {
      switch( error )
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
    do                                                                  \
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

void initGLFW()
{
	static bool alreadyInitialized = false;
	if (alreadyInitialized) return;
	if (!glfwInit())
	exit(EXIT_FAILURE);
	alreadyInitialized = true;
}

#endif

class WindowApp{
public:
	WindowApp(const char title[], vec2i size,
			bool visible = true, bool enableVsync = true);

	virtual void resize(const vec2i &size);

	virtual void render();

	virtual void draw();

  void run();

private:
	vec2i fbSize{0};
	GLuint   fbTexture  {0};
	cudaGraphicsResource_t cuDisplayTexture { 0 };
	uint32_t *fbPointer { nullptr };
	
	/*! the glfw window handle */
	GLFWwindow *handle { nullptr };
	vec2i lastMousePos = { -1,-1 };

	bool resourceSharingSuccessful;
};

NAMESPACE_KRR_END