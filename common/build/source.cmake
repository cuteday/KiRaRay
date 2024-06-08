CONFIGURE_FILE (${KRR_RENDER_SOURCE_DIR}/core/config.in.h ${KRR_RENDER_BINARY_DIR}/include/config.h)

SET ( KRR_INCLUDE_ALL
	${KRR_RENDER_SOURCE_DIR}
	${KRR_RENDER_SOURCE_DIR}/core
	${KRR_RENDER_BINARY_DIR}/include
	${KRR_MATH_INCLUDE_DIRS}
	${KRR_EXT_INCLUDES}
	${CUDA_INCLUDE_DIRS}
	${OptiX_INCLUDE_DIR}
)

SET ( KRR_CORE_SOURCE
	${KRR_RENDER_SOURCE_DIR}/core/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/scenenode.cpp
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
	${KRR_RENDER_SOURCE_DIR}/render/color.cpp
	${KRR_RENDER_SOURCE_DIR}/render/spectrum.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/assimp.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/pbrt.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/openvdb.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/krrscene.cpp
	${KRR_RENDER_SOURCE_DIR}/main/renderer.cpp
	${KRR_RENDER_SOURCE_DIR}/util/tables.cpp
	${KRR_RENDER_SOURCE_DIR}/util/volume.cpp
	${KRR_RENDER_SOURCE_DIR}/util/image.cpp
)

SET (KRR_SOURCE
	${KRR_SOURCE}
	${KRR_RENDER_SOURCE_DIR}/data/rgbspectrum_srgb.cpp
	${KRR_RENDER_SOURCE_DIR}/data/rgbspectrum_aces.cpp
	${KRR_RENDER_SOURCE_DIR}/data/rgbspectrum_dci_p3.cpp
	${KRR_RENDER_SOURCE_DIR}/data/rgbspectrum_rec2020.cpp
	${KRR_RENDER_SOURCE_DIR}/data/named_spectrum.cpp
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

# some files are set to be compiled by nvcc so cuda can resolve 
# the symbols that are defined within these .cpp files.
SET_SOURCE_FILES_PROPERTIES(
	${KRR_RENDER_SOURCE_DIR}/core/camera.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/context.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/optix.cpp
	${KRR_RENDER_SOURCE_DIR}/core/device/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/light.cpp
	${KRR_RENDER_SOURCE_DIR}/core/mesh.cpp
	${KRR_RENDER_SOURCE_DIR}/core/python/py.cpp
	${KRR_RENDER_SOURCE_DIR}/core/renderpass.cpp
	${KRR_RENDER_SOURCE_DIR}/core/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/scenegraph.cpp
	${KRR_RENDER_SOURCE_DIR}/core/texture.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/scene.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/textureloader.cpp
	${KRR_RENDER_SOURCE_DIR}/core/vulkan/uirender.cpp
	${KRR_RENDER_SOURCE_DIR}/core/window.cpp
	
	${KRR_RENDER_SOURCE_DIR}/scene/assimp.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/krrscene.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/openvdb.cpp
	${KRR_RENDER_SOURCE_DIR}/scene/pbrt.cpp

	${KRR_RENDER_SOURCE_DIR}/util/tables.cpp
	
	${KRR_RENDER_SOURCE_DIR}/main/renderer.cpp
	${KRR_RENDER_SOURCE_DIR}/main/kiraray.cpp

	PROPERTIES LANGUAGE CUDA
)

SET_SOURCE_FILES_PROPERTIES (
	${KRR_RENDER_SOURCE_DIR}/render/bdpt/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/render/color.cpp
	${KRR_RENDER_SOURCE_DIR}/render/media.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/denoise/denoise.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/errormeasure/errormeasure.cpp
	${KRR_RENDER_SOURCE_DIR}/render/passes/gbuffer/gbuffer.cpp
	${KRR_RENDER_SOURCE_DIR}/render/path/pathtracer.cpp
	${KRR_RENDER_SOURCE_DIR}/render/profiler/profiler.cpp
	${KRR_RENDER_SOURCE_DIR}/render/profiler/ui.cpp
	${KRR_RENDER_SOURCE_DIR}/render/rasterize/bindless.cpp
	${KRR_RENDER_SOURCE_DIR}/render/spectrum.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/integrator.cpp
	${KRR_RENDER_SOURCE_DIR}/render/wavefront/medium.cpp
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

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/render/wavefront/basic_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/render/wavefront/basic.soa > ${KRR_RENDER_BINARY_DIR}/include/render/wavefront/basic_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/render/wavefront/basic.soa)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/render/bdpt/workitem.soa > ${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/render/bdpt/workitem.soa)


set (KRR_SOA_GENERATED 
	${KRR_RENDER_BINARY_DIR}/include/render/wavefront/workitem_soa.h
	${KRR_RENDER_BINARY_DIR}/include/render/wavefront/basic_soa.h
	${KRR_RENDER_BINARY_DIR}/include/render/bdpt/workitem_soa.h
)

###############################################################################
# generating PTX code from optix shader routines
###############################################################################

INCLUDE_DIRECTORIES (${KRR_INCLUDE_ALL})
INCLUDE (${KRR_RENDER_ROOT}/common/build/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code
CUDA_COMPILE_EMBED(GBUFFER_PTX ${KRR_SHADER_REL_DIR}render/passes/gbuffer/device.cu krr-gbuffer krr_soa_generated) 
CUDA_COMPILE_EMBED(PATHTRACER_PTX ${KRR_SHADER_REL_DIR}render/path/device.cu krr-path krr_soa_generated)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX ${KRR_SHADER_REL_DIR}render/wavefront/device.cu krr-wavefront krr_soa_generated)
CUDA_COMPILE_EMBED(BDPT_PTX ${KRR_SHADER_REL_DIR}render/bdpt/device.cu krr-bdpt krr_soa_generated)

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
)

SET_SOURCE_FILES_PROPERTIES (
	${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/integrator.cpp
	PROPERTIES LANGUAGE CUDA
)

add_custom_command (OUTPUT ${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
    COMMAND soac ${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/guideditem.soa > ${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
    DEPENDS soac ${KRR_RENDER_SOURCE_DIR}/misc/render/ppg/guideditem.soa)

set (KRR_SOA_GENERATED 
	${KRR_SOA_GENERATED}
	${KRR_RENDER_BINARY_DIR}/include/ppg/guideditem_soa.h
)

set(KRR_INCLUDE_ALL
	${KRR_INCLUDE_ALL}
	${KRR_RENDER_SOURCE_DIR}/misc/render
)

# MISC
CUDA_COMPILE_EMBED(PPG_PTX ${KRR_SHADER_REL_DIR}misc/render/ppg/device.cu krr-ppg krr_soa_generated)

SET(KRR_PTX_FILES # MISC
	${KRR_PTX_FILES}
	${PPG_PTX}
	${ZEROGUIDING_PTX}
)
ENDIF()