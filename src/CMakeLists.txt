# a useful blog (for me) about cmake variables and scopes!
# https://www.mgaudet.ca/technical/2017/8/31/some-notes-on-cmake-variables-and-scopes

ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(cuda)

###############################################################################
# include files
###############################################################################

MESSAGE ("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")
MESSAGE ("Optix include directory: ${OptiX_INCLUDE}")

CONFIGURE_FILE (core/config.in.h ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)

SET ( KRR_INCLUDE_ALL
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/core
	${CMAKE_CURRENT_BINARY_DIR}/include
	${ASSIMP_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${IMAGEIO_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE}
)

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
	core/device/optix.h
	core/device/cuda.h
	# non core
	render/path/pathtracer.h
	render/wavefront/integrator.h
	render/wavefront/workitem.h
	render/wavefront/workqueue.h
	render/wavefront/backend.h
	scene/importer.h
)

SET(KRR_HEADERS
	main/renderer.h
	util/check.h
	util/string.h
)

SET (KRR_SOURCE
	render/path/pathtracer.cpp
	render/wavefront/integrator.cpp
	render/wavefront/backend.cpp
	scene/importer.cpp
)

SET(KRR_SHADER_MATERIALS
	render/materials/bxdf.h
	render/materials/matutils.h
	render/materials/diffuse.h
	render/materials/fresnel.h
	render/materials/microfacet.h
	render/materials/disney.h
	render/materials/falcor.h
	render/materials/fresnelblend.h
)

SET ( KRR_SHADERS
	${KRR_SHADER_MATERIALS}
	render/shared.h
	render/shading.h
	render/sampling.h
	render/bsdf.h
	render/lightsampler.h
	render/path/path.h
	render/wavefront/wavefront.h
	render/postprocess.h
	render/path/path.cu
	render/wavefront/wavefront.cu
	render/postprocess.cu
)

FILE ( GLOB KRR_SOURCE_ALL
	main/*
)

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################
add_executable(soac util/soac.cpp)
add_executable (krr::soac ALIAS soac)

target_compile_options(soac PUBLIC ${CMAKE_CXX_FLAGS})
set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa)
set (KRR_SOA_GENERATED ${KRR_SOA_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/workitem_soa.h)

add_custom_target (krr_soa_generated DEPENDS ${KRR_SOA_GENERATED})

###############################################################################
# linking and executables
###############################################################################

INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code 
CUDA_COMPILE_EMBED(PATHTRACER_PTX render/path/path.cu krr-path)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX render/wavefront/wavefront.cu krr-wavefront)

SET(KRR_LIBS_ALL
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
)

ADD_LIBRARY ( krr_lib STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_HEADERS}
	${KRR_SOURCE_ALL}
	${KRR_SHADERS}
	# PTX 
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	# SOA
	${KRR_SOA_GENERATED}
)
ADD_DEPENDENCIES (krr_lib krr_soa_generated)

TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_LIBS_ALL})

ADD_EXECUTABLE ( kiraray main/kiraray.cpp )

TARGET_LINK_OPTIONS (kiraray PUBLIC "/FORCE:MULTIPLE" "/IGNORE:4006")
TARGET_LINK_LIBRARIES (kiraray PUBLIC krr_lib )