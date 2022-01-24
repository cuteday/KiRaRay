ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(cuda)

###############################################################################
# include files
###############################################################################

message("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

INCLUDE_DIRECTORIES ( 
	core
	${CMAKE_CURRENT_SOURCE_DIR}
	${ASSIMP_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE}
)

message("optix include: ${OptiX_INCLUDE}")

SET ( KRR_CORE_SOURCE
	core/window.cpp
	core/light.cpp
	core/scene.cpp
	# components
	core/logger.cpp
	# non core
	main/renderer.cpp
)

SET ( KRR_CORE_HEADERS
	core/kiraray.h
	core/window.h
	core/common.h
	# components
	core/camera.h
	core/scene.h
	core/logger.h

	# maths
	core/math/math.h
	core/math/constants.h
	core/math/vec.h
	core/math/vec/compare.h
	core/math/vec/rotate.h
	core/math/vec/functors.h
	# cuda/optix wrapper
	core/gpu/buffer.h
	core/gpu/optix7.h
	# non core
	main/renderer.h
)

SET ( KRR_SHADERS
	shaders/LaunchParams.h
	shaders/shader.cu
	shaders/shared.h
)

FILE ( GLOB KRR_SOURCE
	#ext/*
	main/*
)

###############################################################################
# linking and executables
###############################################################################

# the argument's name must match the extern variable declared in host c++ code 
cuda_compile_and_embed(PTX_CODE shaders/shader.cu)

ADD_LIBRARY ( krr STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_SHADERS}
	${PTX_CODE}
)

TARGET_LINK_LIBRARIES(krr PUBLIC ${optix_LIBRARY})

SET(ALL_KRR_LIBS
	krr
	krr-cuda
	assimp
	OpenGL::GL
	glfw
	imgui
	${CUDA_LIBRARIES}
)

ADD_EXECUTABLE ( kiraray main/kiraray.cpp )
TARGET_LINK_LIBRARIES (kiraray PUBLIC ${ALL_KRR_LIBS} )