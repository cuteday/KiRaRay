ADD_SUBDIRECTORY(${KRR_RENDER_SOURCE_DIR}/ext)
ADD_SUBDIRECTORY(${KRR_RENDER_SOURCE_DIR}/core/math)

###############################################################################
# include files
###############################################################################
MESSAGE ("Source directory: ${KRR_RENDER_SOURCE_DIR}")
MESSAGE ("Build output directory: ${CMAKE_BINARY_DIR}")
MESSAGE ("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
MESSAGE ("Optix include directory: ${OptiX_INCLUDE_DIR}")
MESSAGE ("Vulkan SDK directory: ${KRR_VULKAN_ROOT}")

INCLUDE (${KRR_RENDER_ROOT}/common/build/source.cmake)

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
	cudart
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
	${KRR_PTX_FILES}
)

SET(KRR_LIBRARIES krr_lib )
ADD_DEPENDENCIES (krr_lib krr_soa_generated)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
SET_PROPERTY(TARGET krr_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_DEFINITIONS (krr_lib PUBLIC ${KRR_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES (krr_lib SYSTEM PUBLIC ${KRR_INCLUDE_ALL} ${OptiX_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(krr_lib PUBLIC ${KRR_SECONDARY_LIBS_ALL})
TARGET_LINK_OPTIONS(krr_lib PUBLIC "LINKER:/NODEFAULTLIB:LIBCMT")

macro(copy_post_build target dll)
 add_custom_command(TARGET ${target} POST_BUILD  
  COMMAND ${CMAKE_COMMAND} -E copy_if_different  
    ${dll}
    $<TARGET_FILE_DIR:${target}>)                 
endmacro()

ADD_SUBDIRECTORY (${KRR_RENDER_SOURCE_DIR}/misc)

IF (KRR_ENABLE_PYTHON)
	pybind11_add_module(pykrr_common ${KRR_RENDER_SOURCE_DIR}/core/python/common.cpp) 
	add_library(pykrr MODULE ${KRR_RENDER_SOURCE_DIR}/core/python/py.cpp)
	TARGET_LINK_LIBRARIES(pykrr PUBLIC ${KRR_LIBRARIES} -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>
		pybind11::module pybind11::lto pybind11::windows_extras)
	pybind11_extension(pykrr)
	set_target_properties(pykrr PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
ENDIF()

ADD_EXECUTABLE ( kiraray ${CMAKE_CURRENT_SOURCE_DIR}/main/kiraray.cpp)
# -WHOLEARCHIVE assures that static variables are instantiated before main(), by linking all object files
TARGET_LINK_LIBRARIES (kiraray PUBLIC ${KRR_LIBRARIES} -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)

copy_post_build(kiraray
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/openvdb/bin/$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>.dll")
copy_post_build(kiraray
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/tbb/lib/$<IF:$<CONFIG:Debug>,debug/tbb_debug.dll,tbb.dll>")
copy_post_build(kiraray "$<TARGET_FILE:IlmBase::Half>")

SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} PARENT_SCOPE)
SET (KRR_LIBRARIES ${KRR_LIBRARIES} PARENT_SCOPE)
