# a useful blog (for me) about cmake variables and scopes!
# https://www.mgaudet.ca/technical/2017/8/31/some-notes-on-cmake-variables-and-scopes

ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(cuda)

###############################################################################
# include files
###############################################################################

MESSAGE ("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")

INCLUDE_DIRECTORIES ( 
	core
	${CMAKE_CURRENT_SOURCE_DIR}
	${ASSIMP_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE}
)

SET (KRR_SHADER_INCLUDE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/core
	${CMAKE_CURRENT_SOURCE_DIR}/shaders
)

message("optix include: ${OptiX_INCLUDE}")

SET ( KRR_CORE_SOURCE
	core/scene.cpp
	core/camera.cpp
	core/light.cpp
	# components
	core/window.cpp
	core/logger.cpp
	# non core
)

SET ( KRR_CORE_HEADERS
	core/kiraray.h
	core/common.h
	# components
	core/logger.h
	core/io.h
	core/window.h
	# rendering
	core/camera.h
	core/scene.h
	core/material.h
	core/sampler.h
	# maths
	core/math/math.h
	core/math/constants.h
	core/math/vec.h
	core/math/mat.h
	core/math/aabb.h
	core/math/quat.h
	core/math/utils.h
	core/math/transform.h
	core/math/vec/compare.h
	core/math/vec/rotate.h
	core/math/vec/functors.h
	# cuda/optix wrapper
	core/gpu/buffer.h
	core/gpu/optix7.h
	# non core
)

SET (KRR_SOURCE
	main/renderer.cu
	main/renderer.cpp
)

SET ( KRR_SHADERS
	shaders/LaunchParams.h
	shaders/shader.cu
	shaders/shared.h
	shaders/path.h
	shaders/postprocess.h
	shaders/postprocess.cu
)

FILE ( GLOB KRR_SOURCE
	#ext/*
	main/*
)

###############################################################################
# linking and executables
###############################################################################

INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code 
CUDA_COMPILE_EMBED(PTX_CODE shaders/shader.cu krr-shader)
#cuda_compile_and_embed(PTX_CODE shaders/shader.cu)

ADD_LIBRARY ( krr STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_SHADERS}
	${PTX_CODE}
)

TARGET_LINK_LIBRARIES(krr PUBLIC ${optix_LIBRARY})

SET(KRR_ALL_LIBS
	krr
	krr-cuda
	#krr-shader
	assimp
	OpenGL::GL
	libglew_static
	glfw
	imgui
	${OPENGL_gl_LIBRARY}
	${CUDA_LIBRARIES}
	${optix_LIBRARY}
)

ADD_EXECUTABLE ( kiraray main/kiraray.cpp )
TARGET_LINK_LIBRARIES (kiraray PUBLIC ${KRR_ALL_LIBS} )