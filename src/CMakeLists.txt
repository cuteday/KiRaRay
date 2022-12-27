ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(core/math)

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
	${KRR_MATH_INCLUDE_DIRS}
	${ASSIMP_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${JSON_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	${IMAGEIO_INCLUDE_DIRS}
	${PBRTPARSER_INCLUDE_DIRS}
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
	core/device/optix.cpp
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
	core/texture.h
	core/sampler.h
	# cuda device backend and host-device interops
	core/interop.h
	core/host/memory.h
	core/host/synchronize.h
	core/host/timer.h
	core/device/atomic.h
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
	render/profiler/fps.h
	render/profiler/ui.h
	scene/importer.h
	scene/scenegraph.h
	# others
	main/renderer.h
	main/vulkan_renderer.h
	util/check.h
	util/string.h
	util/image.h
	util/lowdiscrepancy.h
	util/math_utils.h
	util/tables.h
	util/hash.h
	util/vmf.h
	util/ema.h
)

FILE (GLOB_RECURSE KRR_RENDER_PASSES
	"render/passes/*.cpp"
	"render/passes/*.cu"
)

SET (KRR_SOURCE
	${KRR_RENDER_PASSES}
	render/path/pathtracer.cpp
	render/bdpt/integrator.cpp
	render/wavefront/integrator.cpp
	render/wavefront/backend.cpp
	render/profiler/profiler.cpp
	render/profiler/ui.cpp
	scene/assimp.cpp
	scene/pbrt.cpp
	main/renderer.cpp
	util/tables.cpp
	util/image.cpp
)

SET(KRR_SHADER_MATERIALS
	render/materials/bxdf.h
	render/materials/matutils.h
	render/materials/diffuse.h
	render/materials/fresnel.h
	render/materials/microfacet.h
	render/materials/disney.h
	render/materials/fresnelblend.h
	render/materials/principled.h
	render/materials/dielectric.h
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
)

SET_SOURCE_FILES_PROPERTIES (
	render/wavefront/integrator.cpp
	render/wavefront/backend.cpp
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
INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
# the argument's name must match the extern variable declared in host c++ code 
CUDA_COMPILE_EMBED(PATHTRACER_PTX render/path/path.cu krr-path krr_soa_generated)
CUDA_COMPILE_EMBED(WAVEFRONT_PTX render/wavefront/wavefront.cu krr-wavefront krr_soa_generated)
CUDA_COMPILE_EMBED(BDPT_PTX render/bdpt/device.cu krr-bdpt krr_soa_generated)

ADD_LIBRARY (KRR_PTX STATIC
	# PTX 
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	${BDPT_PTX}
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
	assimp
	cuda 
	cublas
	OpenGL::GL
	libglew_static
	glfw
	imgui
	pbrtParser
	zlibstatic
	${OPENGL_gl_LIBRARY}
	${CUDA_LIBRARIES}
	Vulkan::Vulkan
	KRR_PTX
	krr_math
	krr_cuda_cfg
	krr_cuda_warning
)

ADD_LIBRARY ( krr_lib STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
	${KRR_HEADERS}
	${KRR_SHADERS}
	# SOA (for dependency need, generate soa file before building libraray)
	${KRR_SOA_GENERATED}
	# PTX
	${PATHTRACER_PTX}
	${WAVEFRONT_PTX}
	${BDPT_PTX}
)

ADD_DEPENDENCIES (krr_lib krr_soa_generated)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_DEFINITIONS (krr_lib PUBLIC ${KRR_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_LIBS_ALL})


IF (KRR_ENABLE_PYTHON)
	ADD_LIBRARY(pykrr SHARED core/python/py.cpp)
	TARGET_INCLUDE_DIRECTORIES(pykrr SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
	TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_lib ${PYTHON_LIBRARIES} pybind11::module)
	pybind11_extension(pykrr)
ENDIF()

ADD_EXECUTABLE ( kiraray main/kiraray.cpp)

#TARGET_LINK_OPTIONS (kiraray PUBLIC "/IGNORE:4006")
# -WHOLEARCHIVE assures that static variables are instantiated before main(), by linking all object files
TARGET_LINK_LIBRARIES (kiraray PUBLIC krr_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} PARENT_SCOPE)

# The additional implementation of algorithms (in the static libraray "krr_starlight_lib") is not built by default
# You can turn this on via CMake options (-DKRR_BUILD_STARLIGHT=ON)
if (KRR_BUILD_STARLIGHT)
	set(KRR_BUILD_STARLIGHT 1)
	# ADD LIBRARY FROM MISC  (a.k.a. kirara startlight)
	ADD_SUBDIRECTORY (misc)
	TARGET_LINK_LIBRARIES (krr_lib PUBLIC krr_starlight_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_starlight_lib>)
	SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} ${KRR_MISC_INCLUDE_ALL} PARENT_SCOPE) # export include dirs 
endif()