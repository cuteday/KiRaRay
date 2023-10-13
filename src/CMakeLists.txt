ADD_SUBDIRECTORY(${KRR_RENDER_SOURCE_DIR}/ext)
ADD_SUBDIRECTORY(${KRR_RENDER_SOURCE_DIR}/core/math)

###############################################################################
# include files
###############################################################################
MESSAGE ("Source directory: ${KRR_RENDER_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")
MESSAGE ("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
MESSAGE ("Optix include directory: ${OptiX_INCLUDE}")

CONFIGURE_FILE (${KRR_RENDER_SOURCE_DIR}/core/config.in.h ${KRR_RENDER_BINARY_DIR}/include/config.h)

SET ( KRR_INCLUDE_ALL
	${KRR_RENDER_SOURCE_DIR}
	${KRR_RENDER_SOURCE_DIR}/core
	${KRR_RENDER_BINARY_DIR}/include
	${KRR_MATH_INCLUDE_DIRS}
	${KRR_EXT_INCLUDES}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE}
)

SET ( KRR_CORE_SOURCE
	${KRR_RENDER_SOURCE_DIR}/core/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/scenegraph.cpp
	${KRR_RENDER_SOURCE_DIR}/core/animation.cpp
	${KRR_RENDER_SOURCE_DIR}/core/camera.cpp
	${KRR_RENDER_SOURCE_DIR}/core/light.cpp
	${KRR_RENDER_SOURCE_DIR}/core/mesh.cpp
	${KRR_RENDER_SOURCE_DIR}/core/window.cpp
	${KRR_RENDER_SOURCE_DIR}/core/logger.cpp
	${KRR_RENDER_SOURCE_DIR}/core/file.cpp
	${KRR_RENDER_SOURCE_DIR}/core/renderpass.cpp
	${KRR_RENDER_SOURCE_DIR}/core/texture.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/gpustd.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/context.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/optix.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/scene.cpp
)

SET (KRR_RENDER_PASSES
	${KRR_RENDER_SOURCE_DIR}/render/passes/accumulate/accumulate.cu
	${KRR_RENDER_SOURCE_DIR}/render/passes/denoise/denoise.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/errormeasure/errormeasure.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/errormeasure/metrics.cu
	${KRR_RENDER_SOURCE_DIR}/render/passes/tonemapping/tonemapping.cu
	${KRR_RENDER_SOURCE_DIR}/render/passes/gbuffer/gbuffer.cpp
)

SET (KRR_SOURCE
	${KRR_RENDER_PASSES}
	${KRR_RENDER_SOURCE_DIR}/render/path/pathtracer.cpp
	${KRR_RENDER_SOURCE_DIR}/render/bdpt/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/medium.cpp
	${KRR_RENDER_SOURCE_DIR}/render/rasterize/bindless.cpp
	${KRR_RENDER_SOURCE_DIR}/render/profiler/profiler.cpp
	${KRR_RENDER_SOURCE_DIR}/render/profiler/ui.cpp
	${KRR_RENDER_SOURCE_DIR}/render/media.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/assimp.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/pbrt.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/openvdb.cpp
	${KRR_RENDER_SOURCE_DIR}/main/renderer.cpp
	${KRR_RENDER_SOURCE_DIR}/util/tables.cpp
	${KRR_RENDER_SOURCE_DIR}/util/volume.cpp
	${KRR_RENDER_SOURCE_DIR}/util/image.cpp
)

SET (KRR_SOURCE_VULKAN
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/binding.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/cuvk.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/descriptor.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/helperpass.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/uirender.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/textureloader.cpp
)

SET_SOURCE_FILES_PROPERTIES (
	# some files are set to be compiled by nvcc so cuda can resolve 
	# the symbols that are defined within these .cpp files.
	${KRR_RENDER_SOURCE_DIR}/core/light.cpp
	${KRR_RENDER_SOURCE_DIR}/render/media.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/medium.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/denoise/denoise.cpp
	${KRR_RENDER_SOURCE_DIR}/util/tables.cpp
	PROPERTIES LANGUAGE CUDA
)

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################
add_executable(soac ${KRR_RENDER_SOURCE_DIR}/util/soac.cpp)
add_executable (krr::soac ALIAS soac)

target_compile_options(soac PUBLIC ${CMAKE_CXX_FLAGS})
set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/render/wavefront/workitem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/render/wavefront/workitem.soa > ${KRR_RENDER_BINARY_DIR}/include/render/wavefront/workitem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/render/wavefront/workitem.soa)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/render/bdpt/workitem.soa > ${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/render/bdpt/workitem.soa)

set (KRR_SOA_GENERATED 
	${KRR_RENDER_BINARY_DIR}/include/render/wavefront/workitem_soa.h
	${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
)

###############################################################################
# generating PTX code from optix shader routines
###############################################################################

INCLUDE_DIRECTORIES (${KRR_INCLUDE_ALL})
INCLUDE (${KRR_RENDER_ROOT}/common/build/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code
CUDA_COMPILE_EMBED(GBUFFER_PTX render/passes/gbuffer/device.cu krr-gbuffer krr_soa_generated) 
CUDA_COMPILE_EMBED(PATHTRACER_PTX render/path/path.cu krr-path krr_soa_generated)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX render/wavefront/wavefront.cu krr-wavefront krr_soa_generated)
CUDA_COMPILE_EMBED(BDPT_PTX render/bdpt/device.cu krr-bdpt krr_soa_generated)

SET(KRR_PTX_FILES
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	${GBUFFER_PTX}
	${BDPT_PTX}
)
###############################################################################
# optional starlight contents
###############################################################################

IF(KRR_BUILD_STARLIGHT)
SET(KRR_SOURCE
	${KRR_SOURCE}
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/treemanip.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/medium.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/medium.cpp
)

SET_SOURCE_FILES_PROPERTIES (
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/medium.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/medium.cpp
	PROPERTIES LANGUAGE CUDA
)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/guideditem.soa > ${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/guideditem.soa)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/zero_guiding/guideditem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/guideditem.soa > ${KRR_RENDER_BINARY_DIR}/include/zero_guiding/guideditem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/misc/render/zero_guiding/guideditem.soa)

set (KRR_SOA_GENERATED 
	${KRR_SOA_GENERATED}
	${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
	${KRR_RENDER_BINARY_DIR}/include/zero_guiding/guideditem_soa.h
)

set(KRR_INCLUDE_ALL
	${KRR_INCLUDE_ALL}
	${KRR_RENDER_SOURCE_DIR}/misc/render
)

# MISC
CUDA_COMPILE_EMBED(PPG_PTX misc/render/ppg/device.cu krr-ppg krr_soa_generated)
CUDA_COMPILE_EMBED(ZEROGUIDING_PTX misc/render/zero_guiding/device.cu krr-zeroguiding krr_soa_generated)

SET(KRR_PTX_FILES # MISC
	${KRR_PTX_FILES}
	${PPG_PTX}
	${ZEROGUIDING_PTX}
)
ENDIF()

ADD_LIBRARY (KRR_PTX STATIC ${KRR_PTX_FILES})
ADD_CUSTOM_TARGET (krr_soa_generated DEPENDS ${KRR_SOA_GENERATED})
ADD_DEPENDENCIES (KRR_PTX krr_soa_generated)

###############################################################################
# linking and executables
###############################################################################
# Link openvdb in a standalone library to avoid cuda-openvdb type intersect and dll-wholearchive contracdict

SET(KRR_SECONDARY_LIBS_ALL
	${KRR_SECONDARY_LIBS_ALL}
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

SET(KRR_LIBRARIES krr_lib )
ADD_DEPENDENCIES (krr_lib krr_soa_generated)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_DEFINITIONS (krr_lib PUBLIC ${KRR_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_SECONDARY_LIBS_ALL})
TARGET_LINK_OPTIONS(krr_lib PUBLIC "LINKER:/NODEFAULTLIB:LIBCMT")

macro(copy_post_build target dll)
 add_custom_command(TARGET ${target} POST_BUILD  
  COMMAND ${CMAKE_COMMAND} -E copy_if_different  
    ${dll}
    $<TARGET_FILE_DIR:${target}>)                 
endmacro()

copy_post_build(krr_lib
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/openvdb/bin/$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>.dll")
copy_post_build(krr_lib
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/tbb/lib/$<IF:$<CONFIG:Debug>,debug/tbb_debug.dll,tbb.dll>")
copy_post_build(krr_lib "$<TARGET_FILE:IlmBase::Half>")
ADD_SUBDIRECTORY (${KRR_RENDER_SOURCE_DIR}/misc)

IF (KRR_ENABLE_PYTHON)
	ADD_LIBRARY(pykrr SHARED ${KRR_RENDER_SOURCE_DIR}/core/python/py.cpp)
	TARGET_INCLUDE_DIRECTORIES(pykrr SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
	TARGET_LINK_LIBRARIES(pykrr PUBLIC ${KRR_LIBRARIES} ${PYTHON_LIBRARIES} pybind11::module -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
	pybind11_extension(pykrr)
ENDIF()

ADD_EXECUTABLE ( kiraray ${CMAKE_CURRENT_SOURCE_DIR}/main/kiraray.cpp)
# -WHOLEARCHIVE assures that static variables are instantiated before main(), by linking all object files
TARGET_LINK_LIBRARIES (kiraray PUBLIC ${KRR_LIBRARIES} -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} PARENT_SCOPE)
SET (KRR_LIBRARIES ${KRR_LIBRARIES} PARENT_SCOPE)
