# a useful blog (for me) about cmake variables and scopes!
# https://www.mgaudet.ca/technical/2017/8/31/some-notes-on-cmake-variables-and-scopes

ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(cuda)

###############################################################################
# include files
###############################################################################

MESSAGE ("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")

CONFIGURE_FILE (core/config.in.h ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)

INCLUDE_DIRECTORIES ( 
	core
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/include
	${ASSIMP_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${IMAGEIO_INCLUDE_DIRS}
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
	core/mesh.cpp
	# host-device side
	core/interop.cpp
	core/device/context.cpp
	# components
	core/window.cpp
	core/logger.cpp
	core/file.cpp
	# non core
	core/texture.cpp
)

SET ( KRR_CORE_HEADERS
	core/kiraray.h
	core/common.h
	# components
	core/logger.h
	core/io.h
	core/window.h
	core/taggedptr.h
	core/file.h
	# rendering
	core/raytracing.h
	core/mesh.h
	core/envmap.h
	core/shape.h
	core/light.h
	core/renderpass.h
	core/camera.h
	core/scene.h
	core/material.h
	core/texture.h
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
	# cuda device backend and host-device interops
	core/interop.h
	core/host/memory.h
	core/device/memory.h
	core/device/context.h
	core/device/buffer.h
	core/device/optix7.h
	# non core
	main/renderer.h
	scene/importer.h
)

SET(KRR_HEADERS
	util/check.h
	util/string.h
)

SET (KRR_SOURCE
	main/renderer.cpp
	scene/importer.cpp
)

SET(KRR_MATERIALS
	render/materials/matutils.h
	render/materials/fresnel.h
	render/materials/microfacet.h
)

SET ( KRR_SHADERS
	${KRR_MATERIALS}
	render/shared.h
	render/shading.h
	render/sampling.h
	render/bsdf.h
	render/lightsampler.h
	render/path.h
	render/postprocess.h
	render/path.cu
	render/postprocess.cu
)

FILE ( GLOB KRR_SOURCE_ALL
	#ext/*
	main/*
)

###############################################################################
# linking and executables
###############################################################################

INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code 
CUDA_COMPILE_EMBED(PTX_CODE render/path.cu krr-shader)
#cuda_compile_and_embed(PTX_CODE render/path.cu)

ADD_LIBRARY ( krr STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_HEADERS}
	${KRR_SOURCE_ALL}
	${KRR_SHADERS}
	${PTX_CODE}
)

TARGET_INCLUDE_DIRECTORIES (krr SYSTEM PUBLIC ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr PUBLIC ${optix_LIBRARY})

SET(KRR_ALL_LIBS
	krr
	krr-cuda
	krr_cuda_cfg
	assimp
	OpenGL::GL
	libglew_static
	glfw
	imgui
	zlibstatic
	${OPENGL_gl_LIBRARY}
	${CUDA_LIBRARIES}
	${optix_LIBRARY}
)

ADD_EXECUTABLE ( kiraray main/kiraray.cpp )

TARGET_LINK_OPTIONS (kiraray PUBLIC "/FORCE:MULTIPLE" "/IGNORE:4006")
TARGET_LINK_LIBRARIES (kiraray PUBLIC ${KRR_ALL_LIBS} )