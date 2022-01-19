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

void initGLFW();

#endif

class WindowApp{
public:
	WindowApp(const char title[], vec2i size,
			bool visible = true, bool enableVsync = true);

    ~WindowApp();

	virtual void resize(const vec2i &size);

    void run();

    virtual void render() = 0;

    virtual void draw();

protected:

	vec2i fbSize{0};
	GLuint   fbTexture  {0};
	cudaGraphicsResource_t cuDisplayTexture { 0 };
	uint32_t *fbPointer { nullptr };
	
	/*! the glfw window handle */
	GLFWwindow *handle { nullptr };
	vec2i lastMousePos = { -1,-1 };

	bool resourceSharingSuccessful = false;
};

NAMESPACE_KRR_END