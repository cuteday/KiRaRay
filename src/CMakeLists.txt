ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(core/math)

###############################################################################
# include files
###############################################################################
MESSAGE ("Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")
MESSAGE ("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
MESSAGE ("Optix include directory: ${OptiX_INCLUDE}")

CONFIGURE_FILE (core/config.in.h ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)

SET ( KRR_INCLUDE_ALL
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/core
	${CMAKE_CURRENT_BINARY_DIR}/include
	${KRR_MATH_INCLUDE_DIRS}
	${KRR_EXT_INCLUDES}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE}
)

SET ( KRR_CORE_SOURCE
	core/scene.cpp
	core/scenegraph.cpp
	core/animation.cpp
	core/camera.cpp
	core/light.cpp
	core/mesh.cpp
	core/interop.cpp
	core/window.cpp
	core/logger.cpp
	core/file.cpp
	core/renderpass.cpp
	core/texture.cpp
	core/media.cpp
	core/device/context.cpp
	core/device/optix.cpp
)

SET (KRR_RENDER_PASSES
	render/passes/accumulate/accumulate.cu
	render/passes/denoise/denoise.cpp
	render/passes/errormeasure/errormeasure.cpp
	render/passes/errormeasure/metrics.cu
	render/passes/tonemapping/tonemapping.cu
	render/passes/gbuffer/gbuffer.cpp
)

SET (KRR_SOURCE
	${KRR_RENDER_PASSES}
	render/path/pathtracer.cpp
	render/bdpt/integrator.cpp
	render/wavefront/integrator.cpp
	render/rasterize/bindless.cpp
	render/profiler/profiler.cpp
	render/profiler/ui.cpp
	scene/assimp.cpp
	scene/pbrt.cpp
	main/renderer.cpp
	util/tables.cpp
	util/image.cpp
)

SET (KRR_SOURCE_VULKAN
	core/vulkan/binding.cpp
	core/vulkan/cuvk.cpp
	core/vulkan/descriptor.cpp
	core/vulkan/scene.cpp
	core/vulkan/helperpass.cpp
	core/vulkan/uirender.cpp
	core/vulkan/textureloader.cpp
)

SET_SOURCE_FILES_PROPERTIES (
	render/wavefront/integrator.cpp
	render/passes/denoise/denoise.cpp
	util/tables.cpp
	PROPERTIES LANGUAGE CUDA
)

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################
add_executable(soac util/soac.cpp)
add_executable (krr::soac ALIAS soac)

target_compile_options(soac PUBLIC ${CMAKE_CXX_FLAGS})
set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/render/wavefront/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/render/wavefront/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/wavefront/workitem.soa)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/bdpt/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/bdpt/workitem.soa)

set (KRR_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/render/wavefront/workitem_soa.h
	${CMAKE_CURRENT_BINARY_DIR}/include/render/bdpt/workitem_soa.h
)

add_custom_target (krr_soa_generated DEPENDS ${KRR_SOA_GENERATED})

###############################################################################
# generating PTX code from optix shader routines
###############################################################################

INCLUDE_DIRECTORIES (${KRR_INCLUDE_ALL})
INCLUDE (${KRR_PROJECT_ROOT}/common/build/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code
CUDA_COMPILE_EMBED(GBUFFER_PTX render/passes/gbuffer/device.cu krr-gbuffer krr_soa_generated) 
CUDA_COMPILE_EMBED(PATHTRACER_PTX render/path/path.cu krr-path krr_soa_generated)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX render/wavefront/wavefront.cu krr-wavefront krr_soa_generated)
CUDA_COMPILE_EMBED(BDPT_PTX render/bdpt/device.cu krr-bdpt krr_soa_generated)

ADD_LIBRARY (KRR_PTX STATIC
	# PTX 
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	${GBUFFER_PTX}
	${BDPT_PTX}
)
ADD_DEPENDENCIES (KRR_PTX krr_soa_generated)

###############################################################################
# linking and executables
###############################################################################
SET(KRR_LIBS_ALL
	cuda 
	cublas
	krr_ext
	${CUDA_LIBRARIES}
	KRR_PTX
	krr_math
	krr_cuda_cfg
	krr_cuda_warning
)

ADD_LIBRARY ( krr_lib STATIC
	${KRR_CORE_SOURCE}
	${KRR_SOURCE}
	${KRR_SOURCE_VULKAN}
	# SOA (for dependency need, generate soa file before building libraray)
	${KRR_SOA_GENERATED}
	# PTX
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	${GBUFFER_PTX}
	${BDPT_PTX}
)

ADD_DEPENDENCIES (krr_lib krr_soa_generated)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_DEFINITIONS (krr_lib PUBLIC ${KRR_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_LIBS_ALL})

ADD_SUBDIRECTORY (misc)

IF (KRR_ENABLE_PYTHON)
	ADD_LIBRARY(pykrr SHARED core/python/py.cpp)
	TARGET_INCLUDE_DIRECTORIES(pykrr SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
	TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_lib ${PYTHON_LIBRARIES} pybind11::module -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
	pybind11_extension(pykrr)
ENDIF()

ADD_EXECUTABLE ( kiraray main/kiraray.cpp)

#TARGET_LINK_OPTIONS (kiraray PUBLIC "/IGNORE:4006")
# -WHOLEARCHIVE assures that static variables are instantiated before main(), by linking all object files
TARGET_LINK_LIBRARIES (kiraray PUBLIC krr_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} PARENT_SCOPE)