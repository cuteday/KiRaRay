ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(cuda)

###############################################################################
# include files
###############################################################################
MESSAGE ("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")
MESSAGE ("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
MESSAGE ("Optix include directory: ${OptiX_INCLUDE}")
MESSAGE ("OpenGL libs: ${OPENGL_gl_LIBRARY}")

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
	core/math/complex.h
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
	core/host/synchronize.h
	core/host/timer.h
	core/device/memory.h
	core/device/context.h
	core/device/buffer.h
	core/device/optix.h
	core/device/cuda.h
	core/device/timer.h
)

SET(KRR_HEADERS
	# render
	render/path/pathtracer.h
	render/wavefront/integrator.h
	render/wavefront/workitem.h
	render/wavefront/workqueue.h
	render/wavefront/backend.h
	render/profiler/profiler.h
	render/profiler/ui.h
	scene/importer.h
	# others
	main/renderer.h
	util/check.h
	util/string.h
	util/lowdiscrepancy.h
	util/tables.h
)

SET (KRR_SOURCE
	render/passes/accumulate.cu
	render/passes/tonemapping.cu
	render/path/pathtracer.cpp
	render/wavefront/integrator.cpp
	render/wavefront/backend.cpp
	render/profiler/profiler.cpp
	render/profiler/ui.cpp
	scene/importer.cpp
	util/tables.cpp
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
	render/shading.cuh
	render/sampling.h
	render/bsdf.h
	render/lightsampler.h
	render/path/path.h
	render/wavefront/wavefront.h
	render/passes/accumulate.h
	render/passes/tonemapping.h
	render/raytracing.cuh
)

SET_SOURCE_FILES_PROPERTIES (
	render/wavefront/integrator.cpp
	render/wavefront/backend.cpp
	util/tables.cpp
	PROPERTIES LANGUAGE CUDA
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

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa)
set (KRR_SOA_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h)

add_custom_target (krr_soa_generated DEPENDS ${KRR_SOA_GENERATED})

###############################################################################
# generating PTX code from optix shader routines
###############################################################################

INCLUDE_DIRECTORIES (${KRR_INCLUDE_ALL})
INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code 
CUDA_COMPILE_EMBED(PATHTRACER_PTX render/path/path.cu krr-path)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX render/wavefront/wavefront.cu krr-wavefront)

ADD_LIBRARY (KRR_PTX STATIC
	# PTX 
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
)
ADD_DEPENDENCIES (KRR_PTX krr_soa_generated)

###############################################################################
# some of the cuda code
###############################################################################
# ADD_LIBRARY (KRR_CUDA_CODE STATIC
# 	core/device/cuda.cu
# )
# SET_PROPERTY(TARGET KRR_CUDA_CODE PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# SET_PROPERTY(TARGET KRR_CUDA_CODE PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

###############################################################################
# linking and executables
###############################################################################
SET(KRR_LIBS_ALL
	krr-cuda
	assimp
	OpenGL::GL
	libglew_static
	glfw
	imgui
	zlibstatic
	${OPENGL_gl_LIBRARY}
	${CUDA_LIBRARIES}
	KRR_PTX
	#KRR_CUDA_CODE
	krr_cuda_cfg
	krr_cuda_warning
)

ADD_LIBRARY ( krr_lib STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_HEADERS}
	${KRR_SOURCE_ALL}
	${KRR_SHADERS}
	# SOA
	${KRR_SOA_GENERATED}
	# PTX
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
)

ADD_DEPENDENCIES (krr_lib krr_soa_generated)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_DEFINITIONS (krr_lib PUBLIC ${KRR_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_LIBS_ALL})

ADD_EXECUTABLE ( kiraray main/kiraray.cpp)

#TARGET_LINK_OPTIONS (kiraray PUBLIC "/IGNORE:4006")
TARGET_LINK_LIBRARIES (kiraray PUBLIC krr_lib)